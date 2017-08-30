import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

#__all__ = []

class LabelTool():
    def __init__(self):
        self.isPlaying = True
        self.current_video = -1
        #Set up GUI
        self.__init__videolist()
        self.__labeler()
        self.__init_status()
        self.__cap_video()
        self.__init_gui()
        


    def __init__videolist(self):
        f = open('videolist.txt', 'r')
        self.videolist = [i[:-1] for i in f.readlines() if len(i) > 1]

    def __labeler(self):
        self.current_frame = 0
        self.total_frame = 0
        self.frames = []
        self.label = []

    def __cap_video(self):
        video_path = self.videolist[self.current_video]
        video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
        print(video_path)
        self.cap = cv2.VideoCapture(video_path)
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(self.total_frame)
        self.label = [-1] * self.total_frame
        self.current_frame = 0
    
        self.frames = [self.__get_frame() for _ in range(self.total_frame)]

    def __init_status(self):
        self.stat_play_or_pause = 0
        self.statText_play_or_pause = ['播放', '暂停']

        self.stat_active_or_idle = 0
        self.statText_active_or_idle = ['分类:无效帧', '分类:有效帧']

    def __init_gui(self):
        self.window = tk.Tk()  #Makes main window
        self.window.wm_title("视频逐帧标注")
        self.window.config(background="#FFFFFF")

        #Slider window (slider controls stage position)
        self.sliderFrame = tk.Frame(self.window, width=300, height=100)
        self.sliderFrame.grid(row = 300, column=0, padx=10, pady=2)

        self.strVar_play_or_pause = tk.StringVar(self.sliderFrame)
        self.strVar_play_or_pause.set(self.statText_play_or_pause[self.stat_play_or_pause])
        self.strVar_active_or_idle = tk.StringVar(self.sliderFrame)
        self.strVar_active_or_idle.set(self.statText_active_or_idle[self.stat_active_or_idle])

        self.frame_videolist = tk.Frame(self.window,height = 800)
        self.frame_videolist.grid(row = 0, column=300, padx=10, pady=2)

        self.label_tip = tk.Label(self.frame_videolist, text='待完成视频')
        self.label_tip.grid(row=0)
        self.list_videolist = tk.Listbox(self.frame_videolist)
        self.list_videolist.insert(0, *[i.split('/')[-1] for i in self.videolist])
        self.list_videolist.grid(row=1)
        self.btn_active_or_idle = tk.Button(self.sliderFrame, textvariable=self.strVar_active_or_idle, command=self.__action_active_or_idle)
        self.btn_active_or_idle.grid(row = 0, column=0, padx=10, pady=2)
        self.btn_play = tk.Button(self.sliderFrame, textvariable=self.strVar_play_or_pause, command=self.__action_play_or_pause)
        self.btn_play.grid(row = 0, column=1, padx=10, pady=2)

        self.btn_first_frame = tk.Button(self.sliderFrame, text="第一帧", command=self.__action_first_frame)
        self.btn_first_frame.grid(row = 0, column=4, padx=10, pady=2)
        self.btn_prev_frame = tk.Button(self.sliderFrame, text="上一帧", command=self.__action_prev_frame)
        self.btn_prev_frame.grid(row = 0, column=5, padx=10, pady=2)
        self.btn_next_frame = tk.Button(self.sliderFrame, text="下一帧", command=self.__action_next_frame)
        self.btn_next_frame.grid(row = 0, column=6, padx=10, pady=2)
        self.btn_last_frame = tk.Button(self.sliderFrame, text="末尾帧", command=self.__action_last_frame)
        self.btn_last_frame.grid(row = 0, column=7, padx=10, pady=2)

        self.lb_current_frame = tk.Label(self.sliderFrame)
        self.lb_current_frame.grid(row = 0, column=8, padx=10, pady=2)
        self.lb_current_frame['text'] = '当前帧:-/-'
        self.lb_current_label = tk.Label(self.sliderFrame)
        self.lb_current_label.grid(row = 0, column=9, padx=10, pady=2)
        self.lb_current_label['text'] = '当前帧分类:-'

        self.btn_save = tk.Button(self.sliderFrame, text='保存并读取下一个文件', command=self.__action_save)
        self.btn_save.grid(row = 0, column=10, padx=10, pady=2)
        

        #Graphics window
        self.imageFrame = tk.Frame(self.window, width=1029, height=300)
        self.imageFrame.grid(row=0, column=0, padx=10, pady=2)
        self.lmain = tk.Label(self.imageFrame)
        self.lmain.grid(row=0, column=0)
        self.label_timeline = tk.Label(self.imageFrame)
        self.label_timeline.grid(row = 1, column=0)

    def __get_frame(self):
        _, frame = self.cap.read()
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.resize(cv2image, None, fx=0.5, fy=0.5)
        return cv2image
        # self.frames.append(cv2image)


    def show_frame(self):
        #_, frame = self.cap.read()
        
        #cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #cv2image = cv2.resize(cv2image, None, fx=0.5, fy=0.5)
        #self.frames.append(cv2image)
        if self.current_frame >= self.total_frame:
            return
        img = Image.fromarray(self.frames[self.current_frame])
        imgtk = ImageTk.PhotoImage(image=img)

        
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.label[self.current_frame] = self.stat_active_or_idle
        self.lb_current_frame['text'] = '当前帧:{}/{}'.format(self.current_frame + 1, self.total_frame)
        self.lb_current_label['text'] = '当前帧分类:{}'.format(self.label[self.current_frame])
        self.current_frame += 1

        img_timeline = np.array(self.label)
        mask = img_timeline == 1
        img_timeline[mask] = 255
        mask = img_timeline == 0
        img_timeline[mask] = 128
        mask = img_timeline == -1
        img_timeline[mask] = 0
        img_timeline = np.vstack([img_timeline]*30)
        img = Image.fromarray(img_timeline, 'L')
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_timeline.imgtk = imgtk
        self.label_timeline.configure(image=imgtk)
        if self.stat_play_or_pause == 1:
            self.lmain.after(1, self.show_frame)

# Actions

    def __action_play_or_pause(self):
        if self.stat_play_or_pause == 1:
            self.stat_play_or_pause = 0
        else:
            self.stat_play_or_pause = 1
            self.show_frame()
        self.strVar_play_or_pause.set(self.statText_play_or_pause[self.stat_play_or_pause])

    def __action_active_or_idle(self):
        if self.stat_active_or_idle == 1:
            self.stat_active_or_idle = 0
        else:
            self.stat_active_or_idle = 1
            # self.show_frame()
        self.strVar_active_or_idle.set(self.statText_active_or_idle[self.stat_active_or_idle])
        self.label[self.current_frame] = self.stat_active_or_idle


    def __action_prev_frame(self):
        self.current_frame = max(0, self.current_frame - 2)
        # self.cap.set(1, self.current_frame)
        if self.stat_play_or_pause == 1:
            self.__action_play_or_pause()
        self.show_frame()
    
    def __action_next_frame(self):
        if self.stat_play_or_pause == 1:
            self.__action_play_or_pause()
        self.show_frame()

    def __action_first_frame(self):
        self.current_frame = 0
        if self.stat_play_or_pause == 1:
            self.__action_play_or_pause()
        self.show_frame()

    def __action_last_frame(self):
        self.current_frame = self.total_frame - 1
        if self.stat_play_or_pause == 1:
            self.__action_play_or_pause()
        self.show_frame()
    
    def __action_save(self):
        video_path = self.videolist[self.current_video]
        video_name = '.'.join(video_path.split('/')[-1].split('.')[:-1])
        final = ''.join(str(x) for x in self.label)
        f = open('labels/{}.txt'.format(video_name), 'w')
        f.write(final)
        f.close
        self.current_video += 1
        self.__cap_video()
        self.show_frame()
        self.list_videolist.delete(0)

    def launch(self):
        self.show_frame()
        self.window.mainloop()

if __name__ == '__main__':
    lt = LabelTool()
    lt.launch()
