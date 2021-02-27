import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches

import sys

from PyQt5.QtWidgets import QSizePolicy,\
    QPushButton, QRadioButton, QHBoxLayout, QButtonGroup, QLabel, QWidget, QSlider,\
    QVBoxLayout, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.left = 20
        self.top = 50
        self.title = 'Projections'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        #Предзадаю знаки углов и коэфициент искажения
        self.sigp = 1
        self.sigt = 1
        self.f_scaled = 0
        self.k_scaled = 0

        self.m = PlotCanvas(self)

        button = QPushButton('Изометрическая\n проекция', self)
        button.clicked.connect(self.f_isometric)

        self.butgr1 = QButtonGroup()
        self.butgr2 = QButtonGroup()

        hbox1 = QHBoxLayout()
        radio1 = QRadioButton('+', self)
        radio1.toggle()
        radio2 = QRadioButton('-', self)
        hbox1.addStretch(1)
        hbox1.addWidget(radio1)
        hbox1.addStretch(1)
        hbox1.addWidget(radio2)
        hbox1.addStretch(1)
        self.butgr1.addButton(radio1, id=1)
        self.butgr1.addButton(radio2, id=2)
        self.butgr1.buttonClicked.connect(self.set_p)


        hbox2 = QHBoxLayout()
        radio3 = QRadioButton('+', self)
        radio3.toggle()
        radio4 = QRadioButton('-', self)
        hbox2.addStretch(1)
        hbox2.addWidget(radio3)
        hbox2.addStretch(1)
        hbox2.addWidget(radio4)
        hbox2.addStretch(1)
        self.butgr2.addButton(radio3, 1)
        self.butgr2.addButton(radio4, 2)
        self.butgr2.buttonClicked.connect(self.set_t)

        vbox = QVBoxLayout()
        self.sld2 = QSlider(Qt.Horizontal, self)
        self.sld2.setMaximum(10000)
        self.sld2.valueChanged.connect(self.k_value)

        self.sld = QSlider(Qt.Horizontal, self)
        self.sld.setMaximum(10000)
        self.sld.valueChanged.connect(self.f_value)
        self.val_labels = QLabel('f = {:1.3f}\nphi = {:1.3f}\ntheta = {:1.3f}'.format(
            self.f_scaled,
            self.m.figure_projection.phi * 180 / np.pi,
            self.m.figure_projection.theta * 180 / np.pi))
        self.val_labels.setAlignment(Qt.AlignCenter)
        vbox.addWidget(QLabel('Коэффициент f искажения по оси z:'), stretch=1)
        vbox.addWidget(self.val_labels, stretch=0)
        vbox.addWidget(self.sld, stretch=0)
        vbox.addWidget(button, stretch=1)
        vbox.addWidget(QLabel('Коэффициент k усечения рёбер:'), stretch=1)
        vbox.addWidget(self.sld2, stretch=0)
        vbox.addWidget(QLabel('Знак угла фи:'), stretch=1)
        vbox.addLayout(hbox1, stretch=5)
        vbox.addWidget(QLabel('Знак угла тета:'), stretch=1)
        vbox.addLayout(hbox2, stretch=5)
        vbox.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addWidget(self.m, stretch=2)
        hbox.addLayout(vbox, stretch=1)

        self.setLayout(hbox)
        self.show()
    #Смена активной кнопки
    #---------------------
    def set_p(self):
        id = self.butgr1.checkedId()
        if id == 1:
            self.sigp = 1
        else:
            self.sigp = -1
        self.redrawer()

    def set_t(self):
        id = self.butgr2.checkedId()
        if id == 1:
            self.sigt = 1
        else:
            self.sigt = -1
        self.redrawer()
    #---------------------
    #Отрисовка изометрической проекции
    def f_isometric(self):
        self.f_scaled = np.sqrt(2/3)
        self.sld.blockSignals(True)
        self.sld.setValue(int(self.f_scaled*10000))
        self.sld.blockSignals(False)
        self.redrawer()

    def k_value(self, value):
        self.k_scaled = float(value) / 10000
        self.redrawer()

    def f_value(self, value):
        self.f_scaled= float(value) / 10000
        self.redrawer()

    def redrawer(self):
        self.m.figure_projection.__init__(f=self.f_scaled, sign_phi=self.sigp, sign_theta=self.sigt, k=self.k_scaled)
        self.m.axes.clear()
        self.val_labels.setText('f = {:1.3f}\nphi = {:1.3f}\ntheta = {:1.3f}'.format(
            self.f_scaled,
            self.m.figure_projection.phi * 180 / np.pi,
            self.m.figure_projection.theta * 180 / np.pi))
        self.m.plot()
        self.show()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100, f=0):

        self.f = f
        #
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        #не влияет
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        #
        self.figure_projection = FigureProj(f=self.f)
        self.plot()

    def plot(self):
        size = self.figure_projection.s
        ax = self.figure.add_subplot(111)
        patch = patches.PathPatch(path=self.figure_projection.figure_path,
                                  facecolor="None",
                                  lw=1)
        ax.add_patch(patch)
        # Рисуются стрелочки осей, связанных с телом
        self.draw_axes(ax)
        ax.set_xlim(-2 * size, 2 * size)
        ax.set_ylim(-2 * size, 2 * size)
        # Выключает оси
        ax.axis('off')
        # Изображение всегда "квадратное",
        # то есть масштаб осей сохраняется
        self.figure.gca().set_aspect('equal', adjustable='box')
        self.draw()

    def draw_axes(self, ax):
        path_axes = self.figure_projection.axes_path
        # Добавляет связанные с телом оси к рисунку: х-зеленый, y-красный, z-синий
        # lw - ширина линий
        colors = ["green", "red", "blue"]
        for i in range(len(path_axes)):
            patch = patches.FancyArrowPatch(path=path_axes[i],
                                            color=colors[i],
                                            lw=1,
                                            arrowstyle="-|>,head_length=2,head_width=2")
            ax.add_patch(patch)

class FigureProj():

    def __init__(self, f=0.8165, sign_phi=1, sign_theta = 1, k=0.0):

        self.k = k

        self.s = 10
        # f - коэффициент искажения по оси z
        self.f = f
        self.sign_phi = sign_phi
        self.sign_theta = sign_theta
        # Для отрисовки фигуры и осей
        self.figure, self.figure_codes = self.get_figure()
        self.axes, self.axes_codes = self.get_axes()
        # Матрица проектирования после поворота
        self.matrix, self.phi, self.theta, self.k = self.get_matrix()
        self.figure_path = self.get_path(self.figure, self.figure_codes)
        self.axes_path = [self.get_path(self.axes[i], self.axes_codes[i]) for i in range(len(self.axes))]

    def get_figure(self):
        s = self.s
        k = self.k
        # Задает фигуру как наборы точек и соответсвующие им команды (codes), обрисовывающие по ним контур
        # Каждой грани соответствуют свои точки и свои контуры
        cube_fig = np.array([
             ((1 - k) * s / 2, 0, 0),
             ((1 + k) * s / 2, 0, 0),
             (s, (1 - k) * s / 2, 0),
             (s, (1 + k) * s / 2, 0),
             ((1 + k) * s / 2, s, 0),
             ((1 - k) * s / 2, s, 0),
             (0, (1 + k) * s / 2, 0),
             (0, (1 - k) * s / 2, 0),
             ((1 - k) * s / 2, 0, 0),

             ((1 - k) * s / 2, 0, 0),
             ((1 + k) * s / 2, 0, 0),
             (s, 0, (1 - k) * s / 2),
             (s, 0, (1 + k) * s / 2),
             ((1 + k) * s / 2, 0, s),
             ((1 - k) * s / 2, 0, s),
             (0, 0, (1 + k) * s / 2),
             (0, 0, (1 - k) * s / 2),
             ((1 - k) * s / 2, 0, 0),

             (0, (1 - k) * s / 2, 0),
             (0, (1 + k) * s / 2, 0),
             (0, s, (1 - k) * s / 2),
             (0, s, (1 + k) * s / 2),
             (0, (1 + k) * s / 2, s),
             (0, (1 - k) * s / 2, s),
             (0, 0, (1 + k) * s / 2),
             (0, 0, (1 - k) * s / 2),
             (0, (1 - k) * s / 2, 0),

             ((1 - k) * s / 2, 0, s),
             ((1 + k) * s / 2, 0, s),
             (s, (1 - k) * s / 2, s),
             (s, (1 + k) * s / 2, s),
             ((1 + k) * s / 2, s, s),
             ((1 - k) * s / 2, s, s),
             (0, (1 + k) * s / 2, s),
             (0, (1 - k) * s / 2, s),
             ((1 - k) * s / 2, 0, s),

            ((1 - k) * s / 2, s, 0),
            ((1 + k) * s / 2, s, 0),
            (s, s, (1 - k) * s / 2),
            (s, s, (1 + k) * s / 2),
            ((1 + k) * s / 2, s, s),
            ((1 - k) * s / 2, s, s),
            (0, s, (1 + k) * s / 2),
            (0, s, (1 - k) * s / 2),
            ((1 - k) * s / 2, s, 0),

            (s, (1 - k) * s / 2, 0),
            (s, (1 + k) * s / 2, 0),
            (s, s, (1 - k) * s / 2),
            (s, s, (1 + k) * s / 2),
            (s, (1 + k) * s / 2, s),
            (s, (1 - k) * s / 2, s),
            (s, 0, (1 + k) * s / 2),
            (s, 0, (1 - k) * s / 2),
            (s, (1 - k) * s / 2, 0)])

        codes = np.array([Path.MOVETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.LINETO,
                  Path.CLOSEPOLY, ] * 6)
        return cube_fig, codes

    def get_axes(self):
        s = self.s
        # Возвращает координатные оси, связанные с исходным телом
        # Необходимы для однозначного определения положения тела
        x_axis = np.array([
            (0, 0, 0),
            (s+1, 0, 0)])
        y_axis = np.array([
            (0, 0, 0),
            (0, s+1, 0)])
        z_axis = np.array([
            (0, 0, 0),
            (0, 0, s+1)])
        axis_codes = ([Path.MOVETO, Path.LINETO, ],[Path.MOVETO, Path.LINETO, ],[Path.MOVETO, Path.LINETO, ])
        return (x_axis,y_axis,z_axis), axis_codes

    def get_matrix(self):
        # Принимает параметры f(по учебнику это fz - коэффициент искажения по оси z)
        # и знаки углов фи и тетта
        # f = 0.8165(корень из 2/3) соответсвует изометрической проекции
        f = self.f
        sig_phi = self.sign_phi
        sig_theta = self.sign_theta
        k=self.k

        theta = np.arcsin(f / np.sqrt(2) * sig_theta)
        phi = np.arcsin(f / np.sqrt(2 - f * f) * sig_phi)

        T = np.array([
            [np.cos(phi),  np.sin(phi) * np.sin(theta), 0, 0],
            [0          ,  np.cos(theta)              , 0, 0],
            [np.sin(phi), -np.cos(phi) * np.sin(theta), 0, 0],
            [0          ,  0                          , 0, 1]])
        return T, phi, theta, k

    def get_path(self, coordinates, codes):

        item_homo = self.homo_coordinates(coordinates)
        item_proj = self.projection(item_homo)
        item_proj_2d = self.projection_2d(item_proj)
        path = Path(item_proj_2d, codes)
        return path

    def homo_coordinates(self, coordinates):
        # Добавляется столбец для получения однородной СК(гомогенной)
        return np.append(coordinates, np.ones((np.size(coordinates, 0), 1)), axis=1)

    def projection(self, h_coordinates):
        T = self.matrix
        # Вычисляется проекция через матричное произведение
        return np.dot(h_coordinates, T)

    def projection_2d(self, h_coordinates):
        # Возвращаются двумерные(убираются 2 последних столбца с h
        # и осью Z, в котором все значения обнулились после проектирования) массивы с точками
        return h_coordinates[:, :-2]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())