# -*- coding:utf-8 -*-


"""
This module defines several reusable Qt dialog types.
"""


import traceback
#
from PySide2 import QtCore, QtWidgets


# #############################################################################
# # DIALOGS
# #############################################################################
class FlexibleDialog(QtWidgets.QDialog):
    """
    Dialog class that allows for OK, Yes/No, and timeout interactions.
    To extend this dialog class, override ``setup_ui_body``, ``on_accept`` and
    ``on_reject``, store the instance and call it with ``show`` or ``exec_``.

    Note that ``setup_ui_body`` is being called IN the constructor, so any
    variables that it may need when extending the class need to be set before
    ``super().__init__`` is called.

    As it can be seen here,
    https://stackoverflow.com/questions/56449605/pyside2-qdialog-possible-bug

    implementing a Dialog in PySide2 is a little tricky. These are some things
    to consider:

    * Do not implement ``accept, reject`` directly. Rather, connect the buttons
      to ``accept, reject``, and then connect the ``accepted, rejected``
      signals to custom methods (in this case ``on_accept, on_reject``).

    * When calling the Dialog from the main window, the dialog must be
      persistently stored as a field of the main window ``i.e. self.d = ...``.
      Otherwise it will not show up. Then it can be called in modal or modeless
      way, as follows: ``XXX.connect(self.d.show), ...(self.d.exec_)``.
    """

    TIMEOUT_LBL_TXT = "Closing in {} seconds..."

    def __init__(self, accept_button_name=None, reject_button_name=None,
                 timeout_ms=None, parent=None):
        """
        :param str accept_button_name: Text to be shown in the accept button.
        :param str reject_button_name: Text to be shown in the reject button.
        :param int timeout_ms: If given, time that the dialog takes to close
          automatically, in milliseconds.
        """
        super().__init__(parent)
        # The body goes into ui_widget
        self.ui_widget = QtWidgets.QWidget()
        self.setup_ui_body(self.ui_widget)
        #
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.addWidget(self.ui_widget)
        # Then comes the (optional) buttons section
        with_a = accept_button_name is not None
        with_r = reject_button_name is not None
        if with_a or with_r:
            button_layout = QtWidgets.QHBoxLayout()
            self.main_layout.addLayout(button_layout)
        #
        if with_a:
            self.accept_b = QtWidgets.QPushButton(accept_button_name)
            button_layout.addWidget(self.accept_b)
            self.accept_b.clicked.connect(self.accept)
            self.accepted.connect(self.on_accept)
            # remove autodefault, if you want default set it explicitly
            self.accept_b.setAutoDefault(False)
        #
        if with_r:
            self.reject_b = QtWidgets.QPushButton(reject_button_name)
            button_layout.addWidget(self.reject_b)
            self.reject_b.clicked.connect(self.reject)
            self.rejected.connect(self.on_reject)
            # remove autodefault, if you want default set it explicitly
            self.reject_b.setAutoDefault(False)
        # Finally the (optional) timeout message
        if timeout_ms is not None:
            assert isinstance(timeout_ms, int), \
                "Timeout miliseconds must be int or None!"
            self.timeout_lbl = QtWidgets.QLabel(
                self.TIMEOUT_LBL_TXT.format(timeout_ms / 1000))
            self.main_layout.addWidget(self.timeout_lbl)
        self.timeout_ms = timeout_ms

    def _set_timer(self):
        """
        """
        if self.timeout_ms is not None:
            QtCore.QTimer.singleShot(self.timeout_ms, self.reject)

    def exec_(self, *args, **kwargs):
        """
        Start the dialog in 'exclusive' way, blocking the rest of the app.
        """
        self._set_timer()
        return super().exec_(*args, **kwargs)

    def show(self, *args, **kwargs):
        """
        Start the dialog in parallel to the rest of the app.
        """
        self._set_timer()
        return super().show(*args, **kwargs)

    # OVERRIDE THESE
    def setup_ui_body(self, widget):
        """
        Populate the widget with your desired contents. The widget will
        be above the buttons.
        """
        pass

    def on_accept(self):
        """
        This method will be called if the user presses the (optional)
        accept button.
        """
        pass

    def on_reject(self):
        """
        This method will be called if the user presses the (optional)
        reject button.
        """
        pass


class InfoDialog(FlexibleDialog):
    """
    A type of dialog that shows a header and body strings.
    """

    # If true, the user can select the text with the mouse
    INTERACT_HEADER = False
    INTERACT_BODY = True

    def __init__(self, header, message, accept_button_name=None,
                 reject_button_name=None, timeout_ms=None, parent=None,
                 print_msg=True,
                 header_style="font-weight: bold; color: black"):
        """
        :param str header: This will be the dialog title
        :param str message: This will go below the title, separated by a line
        :param header_style: A CSS-like style to format the header.

        Check ``FlexibleDialog`` docstrings for more details.
        """
        self._h_txt = header
        self._msg_txt = message
        self.print_msg = print_msg
        self._h_style = header_style
        super().__init__(accept_button_name, reject_button_name, timeout_ms,
                         parent)

    def setup_ui_body(self, widget):
        """
        """
        lyt = QtWidgets.QVBoxLayout(widget)
        #
        self.header_lbl = QtWidgets.QLabel(self._h_txt)
        self.body_lbl = QtWidgets.QLabel(self._msg_txt)
        self.line = QtWidgets.QFrame()
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        #
        lyt.addWidget(self.header_lbl)
        lyt.addWidget(self.line)
        lyt.addWidget(self.body_lbl)
        if self.INTERACT_HEADER:
            self.header_lbl.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse)
        if self.INTERACT_BODY:
            self.body_lbl.setTextInteractionFlags(
                QtCore.Qt.TextSelectableByMouse)
        #
        self.header_lbl.setStyleSheet(self._h_style)

    def _print_if(self):
        """
        """
        if self.print_msg:
            print(self._h_txt)
            print(self._msg_txt)

    def exec_(self, *args, **kwargs):
        """
        """
        outcome = super().exec_(*args, **kwargs)
        self._print_if()
        return outcome

    def show(self, *args, **kwargs):
        """
        """
        outcome = super().show(*args, **kwargs)
        self._print_if()
        return outcome


class ExceptionDialog(InfoDialog):
    """
    This class is intended to be used at the main loop level,
    to catch any exceptions that the app may have and show them
    in a Dialog. To do that, it suffices to put the following
    line anywhere before ``app.exec_()``::

      sys.excepthook = ExceptionDialog.excepthook

    Source: https://stackoverflow.com/a/55819545/4511978
    """
    DEACTIVATE = False

    ERROR_TXT = """
ERROR!
If you think this is an app error consider reporting it at:
https://github.com/andres-fr/iamusica_demo/issues"""

    def __init__(self, error_msg, timeout_ms=None, parent=None):
        """
        Check ``InfoDialog`` docstring for details.
        """
        super().__init__(self.ERROR_TXT, error_msg, "OK",
                         "Don't show errors again",
                         header_style="font-weight: bold; color: red")

    def on_reject(self):
        """
        If the user presses on don't show errors again, the whole class
        gets deactivated, so further created instances won't pop up.
        """
        print("deactivated!")
        self.__class__.DEACTIVATE = True

    @classmethod
    def excepthook(cls, exc_type, exc_value, exc_tb):
        """
        Set this method as ``sys.excepthook = <THIS_CLASS>.excepthook``
        somewhere before ``app.exec_()`` to wrap all Python exceptions with
        this dialog.
        """
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(cls.ERROR_TXT)
        print(msg)
        #
        if not cls.DEACTIVATE:
            cls(msg).exec_()
