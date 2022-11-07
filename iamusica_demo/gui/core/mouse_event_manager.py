# -*- coding:utf-8 -*-


"""
This module contains a convenience mixin that provides the following
functionality for mouse tracking:

* Record previous mouse states
* Demultiplex mouse events and preprocess relevant informations

To make a widget responsive to mouse events, simply this class there
and override the desired methods.
"""


from PySide2 import QtCore


class MouseEventManager:
    """
    Extend this class and then instantiate it once as a property in your
    desired widget.

    .. warning::
      The Mixin is compatible with multiple inheritance, but not all
      initializations work. The following does:

      class A(MouseEventManager, QtWidgets.XXX):
        def __init__(self, ...):
          QtWidgets.XXX.__init__(self, ...)
          MouseEventManager.__init__(self, ...)

    In this example, class ``A`` will respond to the overriden
    ``on_move``, etc... methods.
    """

    def __init__(self, track=True):
        """
        :param track: If false, mouseMoveEvents will only be triggered if a
          button is pressed.
        """
        # super().__init__(parent=parent)
        #
        self.setMouseTracking(track)
        self.mousepos_pix = None
        #
        self.last_left_press_pos = None
        self.last_mid_press_pos = None
        self.last_right_press_pos = None
        #
        self.last_left_release_pos = None
        self.last_mid_release_pos = None
        self.last_right_release_pos = None
        #
        self.last_move_pos = None

    def mousePressEvent(self, event):
        """
        Wheel click event handler
        """
        b = event.button()
        if b == QtCore.Qt.MouseButton.LeftButton:
            self.last_left_press_pos = event.pos()
            self.on_left_press(event)
        elif b == QtCore.Qt.MouseButton.MidButton:
            self.last_mid_press_pos = event.pos()
            self.on_mid_press(event)
        elif b == QtCore.Qt.MouseButton.RightButton:
            self.last_right_press_pos = event.pos()
            self.on_right_press(event)

    def mouseReleaseEvent(self, event):
        """
        Wheel release event handler
        """
        b = event.button()
        if b == QtCore.Qt.MouseButton.LeftButton:
            self.last_left_release_pos = event.pos()
            self.on_left_release(event)
        elif b == QtCore.Qt.MouseButton.MidButton:
            self.last_mid_release_pos = event.pos()
            self.on_mid_release(event)
        elif b == QtCore.Qt.MouseButton.RightButton:
            self.last_right_release_pos = event.pos()
            self.on_right_release(event)

    def mouseMoveEvent(self, event):
        """
        """
        p = event.pos()
        # this does happen! filter out
        if p == self.last_move_pos:
            return
        # continue if we had true movement
        bbb = event.buttons()
        # button codes: https://doc.qt.io/qt-5/qt.html#MouseButton-enum
        has_l = bool(bbb & QtCore.Qt.LeftButton)
        has_m = bool(bbb & QtCore.Qt.MidButton)
        has_r = bool(bbb & QtCore.Qt.RightButton)
        #
        self.on_move(event, has_l, has_m, has_r, p, self.last_move_pos)
        self.last_move_pos = p

    def wheelEvent(self, event):
        """
        Override me. This is a simple wrapper, but may include functionality
        like storing positions if needed.
        """
        mods = event.modifiers()
        has_ctrl = bool(mods & QtCore.Qt.ControlModifier)
        has_alt = bool(mods & QtCore.Qt.AltModifier)
        has_shift = bool(mods & QtCore.Qt.ShiftModifier)
        self.on_wheel(event, has_ctrl, has_alt, has_shift)

    def on_left_press(self, event):
        """
        Override me
        """
        pass

    def on_mid_press(self, event):
        """
        Override me
        """
        pass

    def on_right_press(self, event):
        """
        Override me
        """
        pass

    def on_left_release(self, event):
        """
        Override me
        """
        pass

    def on_mid_release(self, event):
        """
        Override me
        """
        pass

    def on_right_release(self, event):
        """
        Override me
        """
        pass

    def on_wheel(self, event, has_ctrl, has_alt, has_shift):
        """
        Override me
        """
        pass

    def on_move(self, event, has_left, has_mid, has_right, this_pos, last_pos):
        """
        Override me
        """
        pass
