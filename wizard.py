import os
from tkinter import *
from tkinter import ttk, filedialog, messagebox, colorchooser
from BaseRoot import Root
models = {'ogi': 'LoadOgi', 'tnd': 'LoadTnd'}

IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
JSON_EXTENSIONS = ['json']
WIDTH = 700
HEIGHT = 200

# helpers functions
def filter_dir_files_by_ext(path: str, extensions: list[str]) -> list[str]:
    return [os.path.join(path, fn) for fn in os.listdir(path)
            if any(fn.lower().endswith(ext) for ext in extensions)]


class Wizard(Root):
    TITLE = 'Cocompare Wizard'

    def __init__(self):
        # self.app_manager = app_manager
        super().__init__(self.TITLE, WIDTH, HEIGHT)
        self.resizable(width=False, height=False)
        self.selected_model = None
        self.images_path = StringVar(name='images')
        self.gt_paths = StringVar(name='gt')
        self.pred_paths = StringVar(name='pred')
        self.params: dict | None = None
        # frames vars for wizard window
        self.frames = None
        self.frame_stack = None
        self.set_wizard()
        self.mainloop()

    def close_wizard(self):
        self.params = {
            'model': self.selected_model,
            'images': self.images_path.get(),
            'gt': self.gt_paths.get(),
            'pred': self.pred_paths.get()
        }
        self.destroy()

    def set_wizard(self):
        self.frames = dict()
        self.frame_stack = []
        self.create_frames()

    def create_frames(self):
        self.frames = {F.__name__: F(self, ) for F in (ModelPicker, LoadTnd, LoadOgi, Finish)}
        self.next_frame('ModelPicker')

    def next_frame(self, frame_name):
        print(f"Switching frame {frame_name}")
        if self.frame_stack and self.frame_stack[-1] is self.frames[frame_name]:
            return
        self.frame_stack.append(self.frames[frame_name])
        self.lift_frame()

    def previous_frame(self):
        if len(self.frame_stack) > 1:
            self.frame_stack.pop(-1)
        self.lift_frame()

    def lift_frame(self):
        self.lower_frames()
        self.frame_stack[-1].lift()

    def lower_frames(self):
        for frame in self.frames.values():
            frame.lower()


class Style(ttk.Style):
    def __init__(self, ):
        super().__init__()
        self.bg_color = 'white'  # background color
        self.fg_color = 'black',  # foreground color
        self.font = ('helvetica', 10)
        self.frame = 'wizard.TFrame'
        self.configure(style=self.frame, background=self.bg_color)
        self.label = 'wizard.TLabel'
        self.configure(style=self.label, background=self.bg_color, font=self.font)
        self.button = 'wizard.TButton'
        self.configure(style=self.button, background=self.bg_color, font=self.font)
        self.radio = 'wizard.TRadiobutton'
        self.configure(style=self.radio, background=self.bg_color, foreground=self.fg_color, font=self.font)


class StepFrame(ttk.Frame):
    def __init__(self, manager, w=WIDTH, h=HEIGHT, **kwargs, ):
        super().__init__(width=w, height=h, **kwargs)
        self.manager = manager
        self.style = Style()
        self.main_frame = self.create_main_frame()
        self.bottom_frame = self.create_bottom_frame()
        self.next_btn = None
        self.configure(style=self.style.frame)
        # self.pack_propagate(False)
        self.pack_propagate(True)  # Allow the frame to resize based on its content
        self.update_min_size()
        # self.pack()
        self.place(x=0, y=0, relwidth=1, relheight=1)

    def update_min_size(self):
        # Force geometry update and set the minsize to content size
        self.update_idletasks()  # Update the geometry manager to reflect widget sizes
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        self.master.winfo_toplevel().minsize(width, height)

    def create_main_frame(self):
        f = ttk.Frame(self, style=self.style.frame)
        f.place(relx=0.5, rely=0.4, anchor='center')
        return f

    def create_bottom_frame(self):
        frame = ttk.Frame(self, width=600, height=30, style=self.style.frame)
        frame.place(relx=1, rely=1, anchor='se')
        return frame

    def button(self, **kwargs):
        btn = ttk.Button(self.bottom_frame, width=10, style=self.style.button, **kwargs)
        btn.pack(side=RIGHT, padx=10, pady=3)
        return btn

    @staticmethod
    def enable_button(btn):
        btn.configure(state='enable')


class Finish(StepFrame):
    def __init__(self, master):
        super().__init__(master)
        self.confirm_var = BooleanVar(value=False)
        self.configure_bottom_buttons()
        self.update_min_size()

    def set_finish_frame(self):
        self.clear_frame()
        self.label(text=f"Model: {self.manager.selected_model}")
        if self.manager.images_path.get():
            self.label(text=f"Images Path: {self.manager.images_path.get()}")
        self.label(text=f"GT Path: {self.manager.gt_paths.get()}")
        self.label(text=f"Prediction Path: {self.manager.pred_paths.get()}")
        confirm_cb = ttk.Checkbutton(self.main_frame, text="Confirm everything is correct",
                                     variable=self.confirm_var, command=self.toggle_finish_button)
        confirm_cb.pack(anchor=W, padx=10, pady=5)

    def label(self, text):
        label = ttk.Label(self.main_frame, text=text)
        label.pack(anchor=W, padx=10, pady=5)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def toggle_finish_button(self):
        self.next_btn.configure(state='normal' if self.confirm_var.get() else 'disabled')

    def configure_bottom_buttons(self):
        self.next_btn = self.button(text='Finish', command=lambda: self.manager.close_wizard(), state='disable')
        self.button(text='Prev', command=self.manager.previous_frame, state='normal')


class LoadModel(StepFrame):
    def __init__(self, master):
        super().__init__(master)
        self.pred_btn = None
        self.gt_btn = None
        self.configure_bottom_buttons()

    def back_to_model_picker(self):
        [var.set('') for var in (self.manager.images_path, self.manager.gt_paths, self.manager.pred_paths)]
        self.manager.next_frame('ModelPicker')

    def configure_bottom_buttons(self):
        self.next_btn = self.button(text='Next', command=self.finish_step, state='disable')
        self.button(text='Prev', command=self.back_to_model_picker, state='normal')

    def finish_step(self):
        self.manager.frames['Finish'].set_finish_frame()
        self.manager.next_frame('Finish')

    def add_entry(self, row: int, var: StringVar, **kwargs):
        btn = ttk.Button(self.main_frame, command=lambda: self.manage_entry(var), **kwargs)
        btn.grid(row=row, column=0, sticky='ew', padx=5, pady=5)
        entry = ttk.Entry(self.main_frame, state='readonly', width=45, textvariable=var)
        entry.grid(row=row, column=1, padx=10, pady=2, sticky=W)
        return btn

    def manage_entry(self, var: StringVar):
        print(var)
        name = var._name
        match name:
            case 'images':
                self.get_images_path()
            case 'gt':
                self.get_coco_path(var, self.pred_btn)
            case 'pred':
                self.get_coco_path(var, self.next_btn)

    def get_images_path(self):
        path = filedialog.askdirectory(title='Select a folder with images')
        if path and self.validate_images_in_directory(path):
            self.manager.images_path.set(path)
            self.gt_btn.configure(state='enable')

    def get_coco_path(self, entry, btn):
        if path := filedialog.askopenfilename(title='Select coco file', filetypes=[("JSON files", "*.json")]):
            entry.set(path)
            btn.configure(state='enable')

    def validate_images_in_directory(self, path):
        if filter_dir_files_by_ext(path, IMAGES_EXTENSIONS):
            return True
        else:
            message = 'The directory you have selected\ndoes not contain any images. Try Again?'
            return self.get_images_path() if messagebox.askyesno('No Images', message) else None


class LoadOgi(LoadModel):
    def __init__(self, master):
        super().__init__(master)
        self.create_files_uploader()

    def create_files_uploader(self):
        self.gt_btn = self.add_entry(row=0, var=self.manager.gt_paths, text='Load Gt Scenes', state='normal')
        self.pred_btn = self.add_entry(row=1, var=self.manager.pred_paths, text='Load Pred Scenes', state='disable')


class LoadTnd(LoadModel):
    def __init__(self, master):
        super().__init__(master)
        self.create_files_uploader()

    def create_files_uploader(self):
        self.add_entry(row=0, var=self.manager.images_path, text='Load Images', state='normal')
        self.gt_btn = self.add_entry(row=1, var=self.manager.gt_paths, text='Load gt', state='disable')
        self.pred_btn = self.add_entry(row=2, var=self.manager.pred_paths, text='Load Predict', state='disable')


class ModelPicker(StepFrame):
    def __init__(self, manager):
        super().__init__(manager)
        self.ogi = ttk.Radiobutton(self)
        self.tnd = ttk.Radiobutton(self)
        self.radio_option = StringVar()
        self.create_model_selector()
        self.configure_bottom_buttons()

    def create_model_selector(self):
        self.radio_option.trace_add("write", self.on_radio_change)

        def create_radio(row: int, col: int, **kwargs):
            radio = ttk.Radiobutton(self.main_frame, style=self.style.radio, **kwargs)
            radio.grid(row=row, column=col, sticky='ew')
            return radio

        self.ogi = create_radio(0, 0, text="OGI", value='ogi', variable=self.radio_option)
        self.ogi = create_radio(1, 0, text="TND", value='tnd', variable=self.radio_option)

    def configure_bottom_buttons(self):
        self.next_btn = self.button(text='Next', command=self.goto_next_frame, state='disable')

    def on_radio_change(self, *e):
        if self.radio_option.get() in ('ogi', 'tnd'):
            self.next_btn.configure(state='normal')

    def goto_next_frame(self):
        opt = self.radio_option.get()
        if opt in ('ogi', 'tnd'):
            self.manager.selected_model = opt
            self.manager.next_frame(models[opt])
        else:
            messagebox.showerror(title='No Model Selected', message='Please select a Model')


if __name__ == '__main__':
    Wizard()
