#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
"""


import numpy as np
from PySide2 import QtCore, QtWidgets, QtGui
#
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as NormalizeColors
#
from ..mouse_event_manager import MouseEventManager
from .core.widgets import NamedForm, IntSpinBox, StrComboBox


# #############################################################################
# ## NUMPY <-> QT_PIXMAP INTERFACING
# #############################################################################
def colorize_matrix(arr, cmap="hot", vmin=None, vmax=None, uint8_output=True,
                    with_alpha=True):
    """
    :param arr: Scalar array of shape ``(h, w)``.
    :param cmap: Matplotlib colormap: matplotlib.org/stable/api/cm_api.html
    :param vmin: If given, normalize color map to this minimum
    :param vmax: If given, normalize color map to this maximum
    :returns: Float array of shape ``(h, w, ch)`` where each hw entry contains
      the colorized from 0 to 1. The number of channels can be 3 or 4, if we
      have alpha: RGB(A).
    """
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    norm = NormalizeColors(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    result = cmap(norm(arr))
    #
    if not with_alpha:
        result = result[:, :, :3].copy()
    #
    if uint8_output:
        result = (result * 255).astype(np.uint8)
    #
    return result


def rgb_arr_to_rgb_pixmap(arr):
    """
    :param arr: Expects a ``np.uint8(h, w, 3)`` array.
    :returns: A ``QtGui.QPixmap`` in format ``RGB888(w, h)``.
    """
    h, w, c = arr.shape
    assert c == 3, "Only np.uint8 arrays of shape (h, w, 3) expected!"
    img = QtGui.QImage(arr.data, w, h,
                       arr.strides[0], QtGui.QImage.Format_RGB888)
    pm = QtGui.QPixmap.fromImage(img)
    return pm


# #############################################################################
# ## SPECTROGRAM
# #############################################################################
class ImageScene(QtWidgets.QGraphicsScene):
    """
    """
    def __init__(self, img_arr=None, parent=None):
        """
        :param img_arr: See ``update_image``
        """
        super().__init__(parent)
        #
        self.img_pmi = None
        self.h = None
        self.w = None
        #
        if img_arr is not None:
            self.update_image(img_arr)

    def update_image(self, img_arr):
        """
        Clears whole scene, and adds the given numpy array as Pixmap.
        :param img_arr: A ``np.uint8(h, w [, ?])`` array.
        """
        pm = rgb_arr_to_rgb_pixmap(img_arr)
        self.clear()
        self.img_pmi = self.addPixmap(pm)
        #
        self.h, self.w = img_arr.shape[:2]
        self.setSceneRect(0, 0, self.w, self.h)

    def get_hw(self):
        """
        """
        if self.img_pmi is not None:
            w, h = self.img_pmi.size().toTuple()
            return (h, w)

    def clear(self):
        """
        """
        super().clear()
        self.img_pmi = None
        self.h, self.w = None, None


class MatrixScene(ImageScene):
    """
    Like ImageScene, but admits ``(h, w)`` matrices and colorizes them to RGB.
    It also stores the matrix for updating purposes
    """
    def __init__(self, parent=None):
        """
        """
        super().__init__(parent=parent)
        self.matrix = None

    def update_image(self, matrix, cmap="hot", vmin=None, vmax=None):
        """
        """
        h, w = matrix.shape  # shape check
        img_arr = colorize_matrix(matrix, cmap, vmin, vmax,
                                  uint8_output=True, with_alpha=False)
        super().update_image(img_arr)
        self.matrix = matrix


class SpectrogramView(QtWidgets.QGraphicsView):
    """
    """
    # Displayed images stick to the left whenever not wide enough
    ALIGNMENT = QtCore.Qt.AlignLeft
    # Allow for vertical stretching
    ASPECT_RATIO = QtCore.Qt.IgnoreAspectRatio

    def __init__(self, display_width=100, parent=None, vert_flip=True):
        """
        """
        super().__init__(parent=parent)
        self.setScene(MatrixScene())
        # super().__init__(ImageScene(), parent=parent)
        # only horiz scroll bar if needed
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # image sticks to the given alignment if too short
        self.setAlignment(self.ALIGNMENT)
        # Width of the view is fixed to this
        self.display_width = display_width
        #
        self.vert_flip = vert_flip
        if vert_flip:
            self.scale(1, -1)

    def fit_h_and_set_xpos(self, xpos=None):
        """
        :param xpos: If given (in pixel ak scene coordinates), view will start
          at this position from the left. If not given, current position is
          rounded to the nearest pixel and maintained.
        """
        if xpos is None:
            # Figure out current xpos and convert to pixel (ak scene) coords
            xpos = int(self.mapToScene(
                self.viewport().geometry()).boundingRect().x())
        scene_h = self.scene().sceneRect().height()
        self.fitInView(QtCore.QRectF(xpos, 0, self.display_width, scene_h),
                       self.ASPECT_RATIO)

    def resizeEvent(self, event):
        """
        This method gets called when the container changes shape. We simply
        adjust the spectrogrogram to fit the new height and maintain xpos.
        """
        self.fit_h_and_set_xpos()


# #############################################################################
# ## PIANO ROLL
# #############################################################################
class RollView(MouseEventManager, SpectrogramView):
    """
    RollView has all functionality from spectrogram, plus analysis functionality
    via interaction with the mouse.
    """
    selRectSignal = QtCore.Signal()
    velGridSignal = QtCore.Signal()



    SELRECT_ZVAL = 2  # 0 is the default. Larger Z goes on top of smaller
    VGRID_ZVAL = 1
    #
    SELRECT_PEN = QtCore.Qt.NoPen  # QtCore.Qt.GlobalColor.yellow
    VGRID_CENTER_STYLE = QtCore.Qt.SolidLine
    VGRID_SIDE_STYLE = QtCore.Qt.DashLine

    def __init__(self, display_width=100, parent=None, vert_flip=True,
                 selrect_rgba=(255, 255, 255, 100),
                 vgrid_center_color=(255, 0, 0, 100),
                 vgrid_side_color=(255, 0, 0, 50),
                 vgrid_width=10):
        """
        :param grid_width: unit is in horizontal view coords (pixels)
        """
        SpectrogramView.__init__(self, display_width, parent, vert_flip)
        MouseEventManager.__init__(self, track=True)
        #
        self.selrect_rgba = selrect_rgba
        self.selrect = None
        #
        self.vgrid_center_color = vgrid_center_color
        self.vgrid_side_color = vgrid_side_color
        self.vgrid_width = vgrid_width
        self.vgrid = None  # a triple of (center_line, [left_lines], [r_l])
        self.selected_notes = []

    def create_vline(self, pos_x, width=0, zval=1,
                     style=QtCore.Qt.SolidLine, rgba=(255, 0, 0, 100)):
        """
        """
        min_y, max_y = 0, self.scene().height()
        vline = QtWidgets.QGraphicsLineItem(pos_x, min_y, pos_x, max_y)
        # set pen (border)
        pen = QtGui.QPen(style)
        pen.setColor(QtGui.QColor(*rgba))
        pen.setWidth(width)
        vline.setPen(pen)
        vline.setZValue(zval)
        #
        return vline

    # vertical grid frontend machinery
    def create_vgrid(self, pos_x, num_sides=None, width=0):
        """
        :param num_sides: How many lines per side around the central position
          (within scene limits). If none give, fills up full scene.
        """
        # if scene is empty don't create item
        if not self.scene().items():
            return
        # delete any preexisting grid
        if self.vgrid is not None:
            self.delete_vgrid()
        self_vgrid = [None, [], []]  # scaffold for self.vgrid
        # get scene boundaries
        min_y, max_y = 0, self.scene().height()
        min_x, max_x = 0, self.scene().width()
        # create+add central line
        vline = self.create_vline(
            pos_x, width, self.VGRID_ZVAL, self.VGRID_CENTER_STYLE,
            self.vgrid_center_color)
        self.scene().addItem(vline)
        self_vgrid[0] = vline
        # create+add leftside lines
        left_pos = np.arange(pos_x, min_x, -self.vgrid_width)[1:][:num_sides]
        for pos in left_pos:
            vline = self.create_vline(
                pos, width, self.VGRID_ZVAL, self.VGRID_SIDE_STYLE,
                self.vgrid_side_color)
            self.scene().addItem(vline)
            self_vgrid[1].append(vline)
        # create+add righttside lines
        right_pos = np.arange(pos_x, max_x, self.vgrid_width)[1:][:num_sides]
        for pos in right_pos:
            vline = self.create_vline(
                pos, width, self.VGRID_ZVAL, self.VGRID_SIDE_STYLE,
                self.vgrid_side_color)
            self.scene().addItem(vline)
            self_vgrid[2].append(vline)
        #
        self.vgrid = self_vgrid
        self.velGridSignal.emit()

    def delete_vgrid(self):
        """
        """
        if self.vgrid is not None:
            # remove central line
            self.scene().removeItem(self.vgrid[0])
            # remove left lines
            while self.vgrid[1]:
                self.scene().removeItem(self.vgrid[1].pop())
            # remove right lines
            while self.vgrid[2]:
                self.scene().removeItem(self.vgrid[2].pop())
        self.vgrid = None

    def mouseDoubleClickEvent(self, event):
        """
        """
        b = event.button()
        x, y = self.mapToScene(event.pos()).toTuple()
        if b == QtCore.Qt.MouseButton.LeftButton:
            # left double click
            self.create_vgrid(x)
        elif b == QtCore.Qt.MouseButton.RightButton:
            # right double click
            self.delete_vgrid()

    # selection rectangle frontend machinery
    def create_selrect(self, pos_x, pos_y):
        """
        """
        # if scene is empty don't create item
        if not self.scene().items():
            return
        # delete any preexisting item
        if self.selrect is not None:
            self.delete_selrect()
        # create new item
        rect = QtWidgets.QGraphicsRectItem(pos_x, pos_y, 0, 0)
        # set brush (surface) and pen (border)
        br = QtGui.QBrush(QtGui.QColor(*self.selrect_rgba))
        rect.setBrush(br)
        pen = QtGui.QPen(self.SELRECT_PEN)
        rect.setPen(pen)
        #
        rect.setZValue(self.SELRECT_ZVAL)
        self.scene().addItem(rect)
        #
        self.selrect = rect

    def delete_selrect(self):
        """
        """
        if self.selrect is not None:
            # get scene (pixel) coords of rectangle, normalized
            x, y, w, h = self.selrect.rect().getRect()
            x0, x1 = sorted((x, x + w))
            y0, y1 = sorted((y, y + h))
            # delete rectangle
            self.scene().removeItem(self.selrect)
            self.selrect = None
            # return normalized coords
            return ((x0, y0), (x1, y1))

    def on_left_press(self, event):
        """
        """
        x, y = self.mapToScene(event.pos()).toTuple()
        self.create_selrect(x, y)

    def on_left_release(self, event):
        """
        """
        if self.selrect is not None:
            ((x0, y0), (x1, y1)) = self.delete_selrect()
            self.on_selrect_released(x0, y0, x1, y1)

    def on_move(self, event, has_left, has_mid, has_right, this_pos, last_pos):
        """
        """
        if self.selrect is not None:
            rect = self.selrect.rect()
            x2, y2 = self.mapToScene(event.pos()).toTuple()
            rect.setBottomRight(QtCore.QPoint(x2, y2))
            self.selrect.setRect(rect)

    # access points to connect with backend
    def get_vgrid_positions(self):
        """
        """
        if self.vgrid is None:
            return None
        #
        xpos_center = self.vgrid[0].line().x1()
        xpos_left = [vline.line().x1() for vline in self.vgrid[1]]
        xpos_right = [vline.line().x1() for vline in self.vgrid[2]]
        return (xpos_center, xpos_left, xpos_right)

    def get_xyv_notes(self, x0, y0, x1, y1):
        """
        :returns: [(x1, y1, v1), ...] where x, y are int indexes of the matrix,
          and v is the corresponding value at each position.
        """
        # if scene is empty don't create item
        if not self.scene().items():
            return
        # ignore "zero-surface" rectangles
        if (x0, y0) == (x1, y1):
            return
        # ignore empty matrix
        matrix = self.scene().matrix
        if matrix is None:
            return
        # get and refine rect boundaries
        h, w = matrix.shape
        xx0, yy0, xx1, yy1 = round(x0), round(y0), round(x1), round(y1)
        xx0, yy0 = max(0, xx0), max(0, yy0)
        xx1, yy1 = min(w, xx1), min(h, yy1)
        # proceed
        matrix_sel = self.scene().matrix[yy0:yy1, xx0:xx1]
        # find nonzero indexes and translate to full matrix
        y_idxs, x_idxs = matrix_sel.nonzero()
        y_idxs, x_idxs = y_idxs + yy0, x_idxs + xx0
        # retrieve values, get notes in xyv format and return
        vals = matrix[y_idxs, x_idxs]
        xyv_list = list(zip(x_idxs, y_idxs, vals))
        return xyv_list

    def on_selrect_released(self, x0, y0, x1, y1):
        """
        Override me
        """
        self.selected_notes = self.get_xyv_notes(x0, y0, x1, y1)
        if self.selected_notes:
            self.selRectSignal.emit()




# #############################################################################
# ## MULTI-SPECTROGRAM
# #############################################################################
class MultiSpectrogramControls(QtWidgets.QWidget):
    """
    """
    FRAME_SUFFIX = " frames"
    DWIDTH_TEXT = "Display width"
    GWIDTH_TEXT = "Grid width"
    SPEC_CMAP_TEXT = "Spectrogram color"
    ROLL_CMAP_TEXT = "Detections color"

    def __init__(self, initial_display_width=1000, initial_vgrid_width=10,
                 parent=None):
        """
        """
        super().__init__(parent=parent)
        lyt = QtWidgets.QHBoxLayout()
        self.setLayout(lyt)
        #
        self.display_width = NamedForm([
            (self.DWIDTH_TEXT,
             lambda default: IntSpinBox(
                 default=initial_display_width, minimum=1,
                 suffix=self.FRAME_SUFFIX), None)])
        lyt.addWidget(self.display_width)
        self.display_width_intbox = self.display_width.form.itemAt(1).widget()
        #
        self.vgrid_width = NamedForm([
            (self.GWIDTH_TEXT,
             lambda default: IntSpinBox(
                 default=initial_vgrid_width, minimum=1,
                 suffix=self.FRAME_SUFFIX), None)])
        lyt.addWidget(self.vgrid_width)
        self.vgrid_width_intbox = self.vgrid_width.form.itemAt(1).widget()
        #
        self.spec_colormap = NamedForm([
            (self.SPEC_CMAP_TEXT,
             lambda default: StrComboBox(items=plt.colormaps()),
             None)])
        self.spec_cmap_cbox = self.spec_colormap.form.itemAt(1).widget()
        lyt.addWidget(self.spec_colormap)
        #
        self.roll_colormap = NamedForm([
            (self.ROLL_CMAP_TEXT,
             lambda default: StrComboBox(items=plt.colormaps()),
             None)])
        self.roll_cmap_cbox = self.roll_colormap.form.itemAt(1).widget()
        lyt.addWidget(self.roll_colormap)

    @staticmethod
    def set_cbox_cmap(cbox, cmap_name):
        """
        """
        cmap_idx = cbox.findText(
            cmap_name, QtCore.Qt.MatchFixedString)
        if cmap_idx >= 0:
            cbox.setCurrentIndex(cmap_idx)

    def get_state(self):
        """
        """
        result = {**self.display_width.get_state(),
                  **self.vgrid_width.get_state(),
                  **self.spec_colormap.get_state(),
                  **self.roll_colormap.get_state()}
        return result


class SpecRollView(QtWidgets.QSplitter):
    """
    Shared x-axis spectrograms
    """
    ORIENTATION = QtCore.Qt.Vertical
    SPEC_VFLIP = True
    ROLL_RANGE = (0, 1)
    SPEC_RANGE = (None, None)
    MIN_DISPLAY_WIDTH = 10

    def __init__(self, parent=None, initial_display_width=800,
                 initial_spec_cmap="bone", initial_roll_cmap="cubehelix",
                 initial_vgrid_width=10):
        """
        """
        super().__init__(parent=parent)
        #
        self.setOrientation(self.ORIENTATION)
        #
        self.spec_view = SpectrogramView(
            display_width=initial_display_width, vert_flip=self.SPEC_VFLIP)
        self.roll_view = RollView(
            display_width=initial_display_width, vert_flip=self.SPEC_VFLIP,
            vgrid_width=initial_vgrid_width, parent=self)
        #
        main_w = QtWidgets.QWidget()
        main_lyt = QtWidgets.QVBoxLayout()
        main_w.setLayout(main_lyt)
        self.addWidget(main_w)
        #
        main_lyt.addWidget(self.spec_view)
        main_lyt.addWidget(self.roll_view)
        # connect all pairs of horizontal scrollbars with each other
        self.spec_view.horizontalScrollBar().valueChanged.connect(
            self.roll_view.horizontalScrollBar().setValue)
        self.roll_view.horizontalScrollBar().valueChanged.connect(
            self.spec_view.horizontalScrollBar().setValue)
        # dynamic display width control
        self.mscontrols = MultiSpectrogramControls(
            initial_display_width=initial_display_width,
            initial_vgrid_width=initial_vgrid_width)
        main_lyt.addWidget(self.mscontrols)
        self._connect_controls_to_views()
        #
        self.mscontrols.set_cbox_cmap(
            self.mscontrols.spec_cmap_cbox, initial_spec_cmap)
        self.mscontrols.set_cbox_cmap(
            self.mscontrols.roll_cmap_cbox, initial_roll_cmap)


    def refresh_all(self, xpos=None):
        """
        Since all sliders are tied, slider behaviour becomes unestable when
        resizing window. This helps stabilizing.
        :param xpos: The single reference to update all positions. If none is
          given, the current xpos of the first spectrogram is used.
        """
        sp0 = self.spec_view
        if xpos is None:
            xpos = int(self.spec_view.mapToScene(
                sp0.viewport().geometry()).boundingRect().x())
        #
        self.spec_view.fit_h_and_set_xpos(xpos)
        self.roll_view.fit_h_and_set_xpos(xpos)

    def resizeEvent(self, event):
        """
        Since all sliders are tied, slider behaviour becomes unestable when
        resizing window. This helps stabilizing.
        """
        # propagate event upwards anyway
        super().resizeEvent(event)
        # but set all spectrograms to the same xpos, regardless of the ordering
        self.refresh_all()

    def clear(self):
        """
        """
        self.spec_view.scene().clear()
        self.spec_view.fit_h_and_set_xpos(xpos=0)
        #
        self.roll_view.scene().clear()
        self.roll_view.fit_h_and_set_xpos(xpos=0)

    def update(self, spec_matrix, roll_matrix, new_xpos="maintain"):
        """
        :param arrays: One per spectrogram
        :new_xpos: Can be 'end' to scroll to the end of the longest updated
          array, 'maintain' to stay in the previous position (unless updated
          arrays are smaller), or a scalar setting the position directly.
        """
        assert spec_matrix.shape[1] == roll_matrix.shape[1], \
            "Spec and roll must have same width!"
        #
        sp0 = self.spec_view
        # get xpos before deleting current
        prev_xpos = int(
            sp0.mapToScene(sp0.viewport().geometry()).boundingRect().x())
        # delete current and replace with new
        self.clear()
        self.spec_view.scene().update_image(spec_matrix)
        self.roll_view.scene().update_image(roll_matrix)
        # but update_image was called without control params, so refresh those
        self.change_spec_appearance()
        self.change_roll_appearance()
        # set xpos to max(0, previous)
        if new_xpos == "maintain":
            new_xpos = min(prev_xpos, spec_matrix.shape[1])
        elif new_xpos == "end":
            new_xpos = max(0, spec_matrix.shape[1] - sp0.display_width)
        else:
            pass  # in this case we assume new_xpos is taken directly
        #
        self.spec_view.fit_h_and_set_xpos(xpos=new_xpos)
        self.roll_view.fit_h_and_set_xpos(xpos=new_xpos)

    # dynamic visualization controls
    def _connect_controls_to_views(self):
        """
        """
        self.mscontrols.display_width_intbox.valueChanged.connect(
            self.change_display_width)
        self.mscontrols.vgrid_width_intbox.valueChanged.connect(
            self.change_vgrid_width)
        self.mscontrols.spec_cmap_cbox.currentTextChanged.connect(
            self.change_spec_appearance)
        self.mscontrols.roll_cmap_cbox.currentTextChanged.connect(
            self.change_roll_appearance)

    def change_display_width(self, display_width):
        """
        """
        self.spec_view.display_width = display_width
        self.roll_view.display_width = display_width
        #
        self.refresh_all()

    def change_vgrid_width(self, vgrid_width):
        """
        """
        self.roll_view.vgrid_width = vgrid_width
        # figure out current grid position
        vgpos = self.roll_view.get_vgrid_positions()
        if vgpos is not None:  # if we do have a grid:
            central_xpos = vgpos[0]
            # if preexisting grid, delete and replace with new width
            self.roll_view.delete_vgrid()
            self.roll_view.create_vgrid(central_xpos)

    def zoom(self, percentage=0):
        """
        """
        ratio = 1.0 + (0.01 * abs(percentage))
        if percentage < 0:
            ratio = 1 / ratio
        current_dw = self.spec_view.display_width
        new_dw = max(self.MIN_DISPLAY_WIDTH, round(current_dw * ratio))
        # this +1 prevents zoom from getting stuck at small value/increment
        if (new_dw == current_dw) and (percentage > 0):
            new_dw += 1
        #
        self.change_display_width(new_dw)

    def change_spec_appearance(self, *args, **kwargs):
        """
        """
        spec_scene = self.spec_view.scene()
        if spec_scene.matrix is not None:
            mscontrols_state = self.mscontrols.get_state()
            #
            spec_scene.update_image(
                spec_scene.matrix,
                cmap=mscontrols_state[self.mscontrols.SPEC_CMAP_TEXT],
                vmin=self.SPEC_RANGE[0], vmax=self.SPEC_RANGE[1])

    def change_roll_appearance(self, *args, **kwargs):
        """
        """
        roll_scene = self.roll_view.scene()
        if roll_scene.matrix is not None:
            mscontrols_state = self.mscontrols.get_state()
            #
            roll_scene.update_image(
                roll_scene.matrix,
                cmap=mscontrols_state[self.mscontrols.ROLL_CMAP_TEXT],
                vmin=self.ROLL_RANGE[0], vmax=self.ROLL_RANGE[1])
