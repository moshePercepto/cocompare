import os
from tkinter import *
from tkinter import ttk, filedialog, colorchooser
from PIL import ImageTk, Image, ImageDraw, ImageFont
from tnd import *
from ogi import *
from BaseRoot import Root
from wizard import Wizard

font = ('helvetica', 10)

VERSION = "1.0.0"
GT_TYPE = "gt"
PREDICT_TYPE = "predict"

rect_thick = 2
font_size = 15
gt_outline: str = "green"
pred_outline: str = "#0c98f0"

IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'bmp', 'png', 'gif']
JSON_EXTENSIONS = ['json']

# model analysis info labels
MODEL_INFO_LABELS = {
    'gt': 'GT:',
    'predict': 'Predict:',
    'tp': 'Hit:',
    'fp': 'False:',
    'fn': 'Miss:',
    'precision': 'Precision:',
    'recall': 'Recall:',
    'f1': 'F1 Score:',
    'iou': 'IoU:',
}
MODEL_GT = 'GT:'
MODEL_PREDICT = 'Predict:'
MODEL_TP = 'Hit:'
MODEL_FP = 'False:'
MODEL_FN = 'Miss:'
MODEL_PRECISION = 'Precision:'
MODEL_RECALL = 'Recall:'
MODEL_F1 = 'F1 Score:'
MODEL_IOU = 'IoU:'


class Viewer:
    def __init__(self, **options):
        self.toggle_visible = False
        self.model = options['model']
        # print(self.model)
        self.modules_parent: OgiLoader | TndLoader = self.set_module_type(**options)

        self.module = self.modules_parent.current
        self.root = Cocompare(self)
        images_path = options['images'] if 'images' in options else options['gt']

        self.paned_window = self.create_paned_window()

        self.list_frame = ListFrame(master=self.root, viewer=self)
        main_frame = ttk.Frame(master=self.root)
        main_frame.pack(fill=BOTH, expand=True)
        self.paned_window.add(main_frame, stretch='always')

        self.root.bind('<Configure>', self.on_resize)

        self.slide = Slide(master=main_frame, viewer=self, images_path=images_path)
        self.bottom_frame = BottomFrame(self)

        self.root.focus_force()
        self.root.mainloop()

    def export_results(self):
        if file_path := filedialog.asksaveasfilename(
                title="name the new file",
                filetypes=[("csv file", "*.csv")],
        ):
            if not file_path.split(".")[-1] == "csv":
                file_path += ".csv"
            self.module.export_data(file_path)

    def create_paned_window(self):
        paned_window = PanedWindow(self.root, orient=HORIZONTAL)
        paned_window.pack(fill=BOTH, expand=True)
        return paned_window

    def toggle_frame(self):
        if self.toggle_visible:
            self.paned_window.forget(self.list_frame)
            # self.slide_frame.resize(self.root.winfo_width())
        else:
            bottom_h = self.bottom_frame.winfo_reqheight()
            self.paned_window.add(self.list_frame, width=2, height=self.root.winfo_height() - bottom_h)
        self.toggle_visible = not self.toggle_visible

    def on_resize(self, event):
        if self.toggle_visible:
            self.paned_window.paneconfig(self.slide, )  # width=self.slide_frame.winfo_width() - 200)
        else:
            self.paned_window.paneconfig(self.slide, )  # width=self.root.winfo_width())

    def set_module_type(self, **options):
        module = OgiLoader(**options) if options['model'] == "ogi" else TndLoader(**options)
        module.build()
        return module

    def update_module_parent(self, index):
        self.modules_parent.index = index
        self.module = self.modules_parent.current
        self.slide.display()


class Cocompare(Root):
    WIDTH = 800
    HEIGHT = 600
    TITLE = f"CoCompare  v_{VERSION}\t"

    def __init__(self, viewer: Viewer, **kwargs):
        super().__init__(self.TITLE, self.WIDTH, self.HEIGHT)
        self.viewer = viewer
        self.config(bg='#222222', bd=0, padx=0, pady=0, highlightthickness=0)
        self.edit_window = None
        self.find_window = None

        self.bind('<Escape>', lambda x: self.quit())
        self.bind('<F11>', self.toggle_window_state)
        self.create_menu()

    def toggle_window_state(self, event=None):
        self.state('zoomed') if self.state() == 'normal' else self.state('normal')

    def create_menu(self):
        menu_bar = Menu(self)
        self.config(menu=menu_bar)
        self.add_settings_menu(menu_bar)

    def add_settings_menu(self, menu_bar):
        settings_menu = Menu(menu_bar, tearoff=0)
        settings_menu.add_command(label="Edit    ctrl+e", command=self.open_edit)
        settings_menu.add_command(label="Find    ctrl+f", command=self.open_find)
        settings_menu.add_separator()
        settings_menu.add_command(label="Export results", command=self.viewer.export_results)
        settings_menu.add_separator()
        settings_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="Options", menu=settings_menu)

    def open_edit(self):
        if not self.edit_window:
            self.edit_window = EditWindow(self.viewer)

    def open_find(self):
        if not self.find_window:
            self.find_window = FindWindow(self.viewer)


class TopWindow(Toplevel):
    def __init__(self, viewer: Viewer, title, **kwargs):
        super(TopWindow, self).__init__(**kwargs)
        self.title(title)
        self.viewer = viewer
        # self.geometry("400x200")
        self.config(bg='#222222', bd=0, padx=0, pady=0, highlightthickness=0)
        self.frame = ttk.Frame(self, padding=(5, 5))
        self.frame.pack(side='top', fill=BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def geometry_centered(self, width, height):
        x_coordinate = int((self.winfo_screenwidth() / 2) - (width / 2))
        y_coordinate = int((self.winfo_screenheight() / 3) - (height / 3))
        self.geometry(f'{width}x{height}+{x_coordinate}+{y_coordinate}')

    def create_combobox(self, var, text: str, row) -> ttk.Combobox:
        values = list(map(str, range(2, 100, 2)))
        label = ttk.Label(self.frame, font=font, text=text, padding=(10, 0))
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        combobox = ttk.Combobox(self.frame, textvariable=var, width=3, values=values)
        combobox.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        combobox.set(rect_thick)  # Set the initial value for the combobox
        return combobox  # Return the combobox widget

    def create_color_button(self, text: str, row, callback, coco_type) -> None:
        color = gt_outline if coco_type == PREDICT_TYPE else pred_outline
        label = ttk.Label(self.frame, text='', padding=(10, 0), width=1, background=color)
        b1 = ttk.Button(self.frame, text=text, command=lambda: callback(coco_type, label))
        b1.grid(row=row, column=0, padx=5, pady=5, sticky="w")
        label.grid(row=row, column=1, padx=5, pady=5, sticky="w")

    def on_close(self):
        if self.title() == 'Find':
            self.viewer.root.find_window = None
        elif self.title() == 'Edit':
            self.viewer.root.edit_window = None
        self.destroy()


class FindWindow(TopWindow):
    def __init__(self, viewer: Viewer, **kwargs):
        super().__init__(viewer=viewer, title="Find", **kwargs)
        self.file_name_entry = None
        self.chr_var = StringVar()
        self.listbox = None
        self.create_window()

    def create_window(self):
        description = 'Please enter a file name or part of it to search for'
        ttk.Label(self.frame, font=font, text=description).grid(row=0)
        entry = ttk.Entry(self.frame, textvariable=self.chr_var, width=70)
        entry.grid(row=1, column=0, padx=5, pady=15)
        entry.bind('<Return>', self.update_listbox)

        btn = ttk.Button(self.frame, text="Find", command=self.update_listbox, width=5)
        btn.grid(row=1, column=1, pady=10, padx=10)

        self.listbox = Listbox(self.frame, height=10, selectmode=SINGLE)
        self.listbox.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")
        # Config listbox to expand when window resize
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create a vertical Scrollbar
        scrollbar = Scrollbar(self.listbox, orient="vertical")
        scrollbar.pack(side=RIGHT, fill=Y)
        # Connect the Scrollbar to the Listbox
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind("<Double-Button-1>", self.update_image)

    def update_listbox(self, e=None):
        chars = self.chr_var.get()
        self.fill_list_box(chars)

    def fill_list_box(self, chars):
        self.listbox.delete(0, END)
        names = self.viewer.module.names_list
        [self.listbox.insert(END, name) for name in names if (chars and chars in name)]

    def update_image(self, event=None):
        pass


class EditWindow(TopWindow):
    def __init__(self, viewer: Viewer, **kwargs):
        super().__init__(viewer=viewer, title="Edit", **kwargs)
        self.gt_color_picker = None
        self.pred_color_picker = None
        self.font_size = None
        self.rect_thick = None

        self.create_edit_options()

    def choose_color(self, coco_type: str, label: ttk.Label):
        global gt_outline, pred_outline
        # variable to store hexadecimal code of color
        if color_code := colorchooser.askcolor(title="Choose color")[1]:
            if coco_type == GT_TYPE:
                gt_outline = color_code
            elif coco_type == PREDICT_TYPE:
                pred_outline = color_code
            label.config(background=str(color_code))
            print(color_code)
        self.lift()

    def create_edit_options(self):
        self.rect_thick = self.create_combobox(rect_thick, 'Rect Thickness:', 0)
        self.font_size = self.create_combobox(font_size, 'Font Size:', 1)

        self.create_color_button("Select GT color", 2, self.choose_color, GT_TYPE)
        self.create_color_button("Select Predict color", 3, self.choose_color, PREDICT_TYPE)

        btn = ttk.Button(self.frame, text="Apply", command=self.update_parameters, width=5, padding=(10, 0))
        btn.grid(row=4, column=0, padx=5, pady=5, sticky="w")

    def update_parameters(self):
        global font_size, rect_thick
        rect_thick = self.rect_thick.get()
        font_size = self.font_size.get()
        self.viewer.slide.set_image()
        self.lift()


class ListFrame(Frame):
    def __init__(self, master, viewer: Viewer):
        super().__init__(master)  # Height to match window
        self.viewer = viewer

        self.path_listbox = self.create_listbox()

        # Button under the Listbox
        self.update_btn = ttk.Button(self, text="Update", command=self.update_path)
        self.update_btn.grid(row=1, column=0, )

        # Configure the listbox frame to adjust with the content
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.path_listbox.bind("<Double-Button-1>", self.update_path)

    def create_listbox(self):
        lb = Listbox(self, selectmode=SINGLE, height=15)
        [lb.insert(END, item) for item in self.viewer.modules_parent.children_list]
        lb.grid(row=0, column=0, sticky=NSEW)
        # Create a vertical Scrollbar
        scrollbar = Scrollbar(lb, orient="vertical")
        scrollbar.pack(side=RIGHT, fill=Y)
        # Connect the Scrollbar to the Listbox
        lb.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=lb.yview)
        return lb

    def update_path(self, event=None):
        # Get the selected path from the Listbox
        selected_index = self.path_listbox.curselection()
        if selected_index:
            self.viewer.update_module_parent(selected_index[0])
            selected_parent = self.path_listbox.get(selected_index)
            print(f"Selected path: {selected_parent}")
        else:
            print("No path selected")


class Slide(Label):
    def __init__(self, master, viewer: Viewer, images_path, **kwargs):
        Label.__init__(self, master, **kwargs)
        self.nav_frame = None
        self.create_navigation_buttons()
        self.prev_btn = None
        self.viewer = viewer
        self.ready = False
        self.images_path = images_path
        self.img_annotations = None
        self.p_img: ImageTk.PhotoImage | None = None
        self.rect_thick = 2
        self.font_size = 15
        self.configure(bg=viewer.root['bg'])
        self.current_image: OgiImage | OgiImage | None = None
        self.image: Image | None = None
        self.image_width = None
        self.image_height = None
        self.image_x_offset = None
        self.image_y_offset = None
        self.scale = 1.0
        self.images_list = None
        self.index = 0
        self.pack(fill='both', expand=True)
        self.bind("<Configure>", self.set_image)
        self.gt = False
        self.predict = False
        # self.get_images_parent_dir()

        self.bind("<Left>", lambda event: self.set_index(-1))
        self.bind("<Right>", lambda event: self.set_index(1))

    def create_navigation_buttons(self):
        self.nav_frame = Frame(self)
        self.nav_frame.place(relx=0.5, rely=0.8, anchor='se')  # Adjust position as needed

        # Create Next button
        next = ttk.Button(self.nav_frame, text="Next", command=lambda event: self.set_index(1))
        next.place(relx=0.9, rely=0.9, anchor="se")  # Bottom-right corner

        # Create Prev button
        prev = ttk.Button(self.nav_frame, text="Prev", command=lambda event: self.set_index(-1))
        prev.place(relx=0.1, rely=0.9, anchor="sw")  # Bottom-left corner

    def search_image_index(self, image_name: str):
        index = self.viewer.module.get_image_index(image_name)
        print(self.index, index)
        self.index = index
        self.set_index(0)

    def set_index(self, num):
        self.index = (self.index + num) % len(self.viewer.module.images)
        self.display()
        self.viewer.bottom_frame.update_info()
        self.nav_frame.lift()

    def display(self):
        def get_image_path():
            if self.viewer.model == 'ogi':
                return os.path.join(self.images_path, self.viewer.module.name, 'images', self.current_image.name)
            return os.path.join(self.images_path, self.current_image.name)

        self.current_image = self.viewer.module.images[self.index]
        img_path = get_image_path()
        if os.path.exists(img_path):
            self.image = Image.open(img_path)
            self.set_image()
        else:
            image = Image.new('RGB', (100, 100), color='white')
            self.config(image=ImageTk.PhotoImage(image))
            # self.set_none_image()
        title = f"{self.viewer.root.TITLE}\t\t{self.viewer.module.name} | {self.current_image.name}"
        self.viewer.root.title(title)

    def get_img_name(self, ):
        return self.current_image.name

    def set_image(self, event=None):
        if self.image:
            print(f"{'=' * 100}\nimage name: {self.get_img_name()}\n")
            img = self.resize_image()
            if self.current_image:
                # bottom_frame.set_info_frame()
                img = self.drew_annotations(img)
            print(f"{'=' * 100}\n")
            self.p_img = ImageTk.PhotoImage(img)
            self.config(image=self.p_img)

    def resize_image(self):
        iw, ih = self.image.width, self.image.height
        mw, mh = self.master.winfo_width(), self.master.winfo_height()
        mh = mh - self.viewer.bottom_frame.winfo_reqheight() - 20
        print(f"{self.viewer.bottom_frame.winfo_reqheight()=}")
        print(f"{ih=}({type(ih)}), {iw=}{type(iw)})")
        if iw > ih:
            ih = ih * (mw / iw)
            r = mh / ih if (ih / mh) > 1 else 1
            iw, ih = mw * r, ih * r
        else:
            iw = iw * (mh / ih)
            r = mw / iw if (iw / mw) > 1 else 1
            iw, ih = iw * r, mh * r
        x_offset = max((mw - iw) // 2, 0)
        y_offset = max((mh - ih) // 2, 0)

        self.image_x_offset = x_offset
        self.image_y_offset = y_offset
        self.image_width = iw
        self.image_height = ih
        # print(f"image resize width: {iw}, height: {ih}")

        return self.image.resize((int(iw * self.scale), int(ih * self.scale)))

    def drew_annotations(self, img):
        gt_bboxes = self.current_image.gt
        pred_bboxes = self.current_image.pred

        def print_iou():
            for gt_bbox in self.current_image.gt:
                gt_bbox = OgiImage.bbox2pt(gt_bbox)
                for pred_bbox in self.current_image.pred:
                    pred_bbox = OgiImage.bbox2pt(pred_bbox)
                    iou = self.current_image.calc_iou(gt_bbox, pred_bbox, 0.6)
                    print(f"{iou=}\n\t{gt_bbox=}\n\t{pred_bbox=}\n")

        print_iou()

        def drew_bboxes(bboxes, outline):
            draw = ImageDraw.Draw(img)
            drew_font = ImageFont.truetype("arial.ttf", int(font_size))
            for bbox in bboxes:
                bbox = self.viewer.slide.calc_bbox(bbox, self.image_width, self.image_height)
                draw.rectangle(bbox, outline=outline, width=int(rect_thick))

        drew_bboxes(gt_bboxes, gt_outline)
        drew_bboxes(pred_bboxes, pred_outline)
        return img

    @staticmethod
    def calc_bbox(bbox, width, height):
        # print(f"before calc_bbox: {bbox}")
        x, y, w, h = bbox
        left = x * width
        top = y * height
        right = left + (w * width)
        bottom = top + (h * height)
        # print(f"after calc_bbox: {[left, top, right, bottom]}")
        return left, top, right, bottom
        # return x1 * width, y1 * height, w, h

    def initiate_slide(self):
        self.display()
        # self.current_image = self.viewer.module.

    # def set_none_image(self):
    #     self.


class BottomFrame(ttk.Frame):

    def __init__(self, viewer: Viewer, **kwargs):
        super().__init__(viewer.root, **kwargs)
        self.viewer = viewer
        # info labels
        self.toggle_btn = None
        self.gt_count = None
        self.pred_count = None
        self.tp = None
        self.fp = None
        self.tn = None
        self.iou = None
        self.recall = None
        self.precision = None
        self.f1 = None

        self.img_gt_count = None
        self.img_pred_count = None
        self.img_tp = None
        self.img_fp = None
        self.img_tn = None

        self.pack(side='bottom', fill='x')
        self.style = ttk.Style()
        self.style.configure(style='bar.TFrame', background='white')
        self.style.configure(style='bar.TButton', background='white', font=font)
        self.style.configure(style='bar.TLabel', background='white', font=font)
        self.configure(style='bar.TFrame')
        self.create_toggle_button()
        self.create_nav_buttons()
        self.create_general_info_frame()
        self.create_image_info_frame()
        self.update_info()

    def create_general_info_frame(self):
        frame = ttk.Frame(self, padding=(5, 5))
        frame.pack(side='left')

        def create_label(text: str):
            label = ttk.Label(frame, font=font, style='bar.TLabel', text=text, padding=(2, 0))
            label.pack(side="left")
            return label

        self.gt_count = create_label(MODEL_GT)
        self.pred_count = create_label(MODEL_PREDICT)
        self.tp = create_label(MODEL_TP)
        self.fp = create_label(MODEL_FP)
        self.tn = create_label(MODEL_FN)
        self.iou = create_label(MODEL_IOU)
        self.precision = create_label(MODEL_PRECISION)
        self.recall = create_label(MODEL_RECALL)
        self.f1 = create_label(MODEL_F1)

    def create_image_info_frame(self):
        frame = ttk.Frame(self, padding=(5, 5))
        frame.pack(side='left')

        def create_label(text: str):
            label = ttk.Label(frame, font=font, style='bar.TLabel', text=text, padding=(2, 0))
            label.pack(side="left")
            return label

        create_label('Image info: ')
        self.img_gt_count = create_label(MODEL_GT)
        self.img_pred_count = create_label(MODEL_PREDICT)
        self.img_tp = create_label(MODEL_TP)
        self.img_fp = create_label(MODEL_FP)
        self.img_tn = create_label(MODEL_FN)

    # def set_info_frame(self):
    #     gt, pred, tp, fp, tn, iou = manager.get_iou_info()
    #     recall, precision, f1= manager.get_iou_analysis()
    #     print(f"gt: {gt}, pred: {pred}, tp: {tp}, fp: {fp}, tn: {tn}")
    #     self.update_info(gt=gt, pred=pred, tn=tn, fp=fp, tp=tp, iou=iou, recall=recall, precision=precision, f1=f1)

    def update_info(self):
        self.update_general_info()
        self.update_image_info()

    def update_image_info(self, ):
        gt, pred, tp, fp, tn = self.viewer.module.images[
            self.viewer.slide.index].get_info() if self.viewer.module.images else (0, 0, 0, 0, 0)
        self.img_gt_count.config(text=f"{MODEL_GT} {gt}")
        self.img_pred_count.config(text=f"{MODEL_PREDICT} {pred}")
        self.img_tp.config(text=f"{MODEL_TP} {tp}")
        self.img_fp.config(text=f"{MODEL_FP} {fp}")
        self.img_tn.config(text=f"{MODEL_FN} {tn}")

    def update_general_info(self, ):
        recall, precision, f1, condition = self.viewer.module.get_analysis()
        self.recall.config(text=f"{MODEL_RECALL} {recall}")
        self.precision.config(text=f"{MODEL_PRECISION} {precision}")
        self.f1.config(text=f"{MODEL_F1} {f1}")

        images_count, gt, pred, tp, fp, fn, th = self.viewer.module.get_info()
        self.gt_count.config(text=f"{MODEL_GT} {gt}")
        self.pred_count.config(text=f"{MODEL_PREDICT} {pred}")
        self.tp.config(text=f"{MODEL_TP} {tp}")
        self.fp.config(text=f"{MODEL_FP} {fp}")
        self.tn.config(text=f"{MODEL_FN} {fn}")
        self.iou.config(text=f"{MODEL_IOU} {condition}")

    def create_nav_buttons(self):
        frame = ttk.Frame(self, padding=(5, 5), )
        frame.pack(side='right', anchor='center')
        ttk.Button(frame, text='< prev', style='bar.TButton', command=lambda: self.viewer.slide.set_index(-1)).pack(
            side="left")
        ttk.Button(frame, text='next >', style='bar.TButton', command=lambda: self.viewer.slide.set_index(1)).pack(
            side="left")

    def create_toggle_button(self):
        print(self.winfo_height())
        self.toggle_btn = ttk.Button(self, text="scenes", style='bar.TButton', command=self.viewer.toggle_frame)
        self.toggle_btn.pack(side=RIGHT, pady=10)


if __name__ == '__main__':
    image_path = r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\images"
    gt_list = [r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\COCO_test_R2T_oldata_BB_issues.json"]
    pred_list = [
        r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\M3T_anomalies_meregd_02.json",
        r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\M30T_anomalies_meregd_02.json",
        r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\ZH20T_anomalies_meregd_02.json"
    ]
    gt = r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\ogi\gt'
    pred = r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\ogi\predictions_2024-09-04-12-43-48.json'
    args = {'model': 'ogi', 'gt': gt, 'pred': pred, 'th': 0.001, 'd_th': 0.02}
    # Wizard()
    Viewer(**args)
    # Initialize root and display widgets
    # tnd_tests = TndTests(gt_list, pred_list, 0.3)
    # manager = tnd_tests.current
    # root = Root()
    # # slide = Slide(root)
    # # bottom_frame = BottomFrame(root)
    # # slide.load_images_dir()
    #
    # # Bind keys for navigation
    # root.bind("<Left>", lambda event: slide.set_index(-1))
    # root.bind("<Right>", lambda event: slide.set_index(1))
    #
    # root.focus_force()
    # root.mainloop()

# ----------------------------------- helper functions ----------------------------------------
