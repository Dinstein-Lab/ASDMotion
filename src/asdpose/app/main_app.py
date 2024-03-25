import shlex
import subprocess
import time
from os import path as osp
import shutil
from tkinter import Tk, PanedWindow, Listbox, BOTH, Button, TOP, END, NORMAL, DISABLED, Checkbutton, BooleanVar
from tkinter.filedialog import askopenfilenames

from asdpose.logger import LogManager
from asdpose.utils import PROJECT_ROOT

logger = LogManager.APP_LOGGER


class Display:
    video_types = [('Video files', '*.avi;*.mp4')]

    def __init__(self):
        self.main_path = osp.join(PROJECT_ROOT, 'src', 'asdpose', 'detector', 'main.py')
        self.python_path = shutil.which('python')
        self.root = Tk()
        self.root.title('ASDPose')
        self.video_paths = []

        self.browse_panel = panel = PanedWindow(self.root, name='browsePanel')
        self.videos_listbox = Listbox(panel, name='videosListbox', width=80, height=4)

        self.browse_button = Button(self.browse_panel, name='browseButton', text="Browse", command=self.browse_button_click)
        self.start_button = Button(self.browse_panel, name='startButton', text='Start', state=DISABLED, command=self.start_button_click)
        self.child_detect = BooleanVar()
        self.child_detect_button = Checkbutton(self.browse_panel, text='Child Detection', variable=self.child_detect)
        self.exec_jordi = BooleanVar()
        self.exec_jordi_button = Checkbutton(self.browse_panel, text='ASDPose', variable=self.exec_jordi)
        # self.exec_barni = BooleanVar()
        # self.exec_barni_button = Checkbutton(self.browse_panel, text='BARNI', variable=self.exec_barni)

        self.videos_listbox.pack(fill=BOTH, expand=1)
        self.browse_button.pack(expand=1)
        self.start_button.pack(expand=1)
        self.exec_jordi_button.pack(expand=1)
        # self.exec_barni_button.pack(expand=1)
        self.child_detect_button.pack(expand=1)
        self.browse_panel.pack(side=TOP, fill=BOTH, expand=1)

    def run(self):
        self.root.mainloop()

    def browse_button_click(self):
        video_paths = askopenfilenames(title='Select video file', filetypes=Display.video_types)
        self.video_paths = self.root.tk.splitlist(video_paths)
        listbox = self.root.nametowidget('browsePanel.videosListbox')
        listbox.delete(0, END)
        for item in self.video_paths:
            listbox.insert(END, item)
        b = len(self.video_paths) > 0
        self.start_button.config(state=NORMAL if b else DISABLED)

    def start_button_click(self):
        for v in self.video_paths:
            logger.info(f'Analyzing video: {v}')
            s = time.time()
            cmd = f'{self.python_path} "{self.main_path}" -video "{v}" -out "{osp.join(PROJECT_ROOT, "resources", "runs")}" -cd {int(self.child_detect.get())}'.replace('\\', '/')
            logger.info(f'Executing: {cmd}')
            subprocess.check_call(shlex.split(cmd), universal_newlines=True)
            logger.info(f'{model} finished successfully.')
            t = time.time()
            delta = t - s
            logger.info(f'Total {int(delta // 3600):02d}:{int((delta % 3600) // 60):02d}:{delta % 60:05.2f} for {v}')


if __name__ == '__main__':
    d = Display()
    d.run()
