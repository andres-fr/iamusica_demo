# -*- coding:utf-8 -*-


"""
This module is a library of reusable, extendable Qt widgets, including:
* Simple widget factories
* Simple widgets
* Composite widgets
* Forms
"""


import os
#
from PySide2 import QtWidgets, QtCore
#
# from .utils import change_label_font, recursive_delete_qt

from .. import change_label_font


# # #############################################################################
# # # SIMPLE WIDGET FACTORIES
# # #############################################################################
# def get_scroll_area(vertical_bar=True, horizontal_bar=True):
#     """
#     This function creates and configures a scroll area. It also creates a
#     dummy widget inside the scroll area, that can be further populated with
#     layouts. The scroll area itself can be added to layouts as a widget.

#     :returns: ``(scroll_area, widget)``
#     """
#     scroller = QtWidgets.QScrollArea()
#     if vertical_bar:
#         scroller.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
#     else:
#         scroller.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#     if horizontal_bar:
#         scroller.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
#     else:
#         scroller.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
#     #
#     scroller.setWidgetResizable(True)
#     scroller_widget = QtWidgets.QWidget()
#     scroller.setWidget(scroller_widget)
#     #
#     return scroller, scroller_widget


# def get_separator_line(horizontal=False):
#     """
#     """
#     line = QtWidgets.QFrame()
#     line.setFrameShape(line.HLine if horizontal else line.VLine)
#     line.setFrameShadow(line.Sunken)
#     return line


# # #############################################################################
# # # BASIC WIDGETS
# # #############################################################################
class WidgetWithValueState:
    """
    Mixin to avoid code repetition. It adds the ``get_state`` getter to the
    class, returning the ``value()`` field. Needed e.g. for widgets used inside
    ``NamdedForm``.
    """
    def get_state(self):
        return self.value()




class StrComboBox(QtWidgets.QComboBox, WidgetWithValueState):
    """
    A NamedForm-compatible ComboBox
    """
    def __init__(self, parent=None, items=None):
        """
        """
        super().__init__(parent=parent)
        if items is not None:
            self.addItems(items)

    def get_state(self):
        """
        """
        return self.currentText()

class IntSpinBox(QtWidgets.QSpinBox, WidgetWithValueState):
    """
    Qt SpinBox contains floats by default. This class is the integer version
    with a convenience interface.
    """
    def __init__(self, parent=None, minimum=-1_000_000, maximum=1_000_000,
                 step=1, default=0, suffix=""):
        """
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setSingleStep(step)
        self.setValue(default)
        if suffix:
            self.setSuffix(suffix)


class DecimalSpinBox(QtWidgets.QDoubleSpinBox, WidgetWithValueState):
    """
    Qt DoubleSpinBox with a convenience interface
    """
    def __init__(self, parent=None, minimum=-1_000_000, maximum=1_000_000,
                 step=1.23, default=0, suffix=""):
        """
        """
        super().__init__(parent)
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self.setSingleStep(step)
        self.setValue(default)
        if suffix:
            self.setSuffix(suffix)
        # infer number of decimals from given step
        num_decimals = str(step)[::-1].find('.')
        self.setDecimals(num_decimals)


class PvalueSpinBox(DecimalSpinBox, WidgetWithValueState):
    """
    """
    def __init__(self, parent, default, step=0.001):
        """
        """
        super().__init__(parent, minimum=0, maximum=1,
                         default=default, step=step)


class DecimalSlider(QtWidgets.QSlider):
    """
    Qt sliders have integer domains by default. This is a decimal adaption.
    Source: https://stackoverflow.com/a/50300848/4511978
    """

    decimalValueChanged = QtCore.Signal(float)

    def __init__(self, minimum=0, maximum=100, decimals=3, *args, **kargs):
        super().__init__( *args, **kargs)
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDecimalValueChanged)
        #
        self.setMinimum(minimum)
        self.setMaximum(maximum)

    def value(self):
        val = float(super().value()) / self._multi
        return val

    def setValue(self, value):
        """
        """
        val = int(value * self._multi)
        super().setValue(val)

    def emitDecimalValueChanged(self):
        """
        """
        self.decimalValueChanged.emit(self.value())

    def setMinimum(self, value):
        """
        """
        return super().setMinimum(value * self._multi)

    def setMaximum(self, value):
        """
        """
        return super().setMaximum(value * self._multi)

    def getMaximum(self):
        """
        """
        return float(super().maximum()) / self._multi

    def setSingleStep(self, value):
        """
        """
        return super().setSingleStep(value * self._multi)

    def singleStep(self):
        """
        """
        val = float(super().singleStep()) / self._multi
        return val



# class BoolCheckBox(QtWidgets.QCheckBox):
#     """
#     """
#     def __init__(self, parent, default=False):
#         """
#         """
#         super().__init__(parent)
#         self.setChecked(default)

#     def get_state(self):
#         """
#         """
#         return bool(self.checkState())


# class PathPrompt(QtWidgets.QWidget):
#     """
#     A text form and a button that opens a file dialog. Choosing a file
#     in the dialog writes its path on the text form. ``get_state`` returns
#     the contents of the text form
#     """

#     BUTTON_TXT = "Load file"

#     def __init__(self, parent, default="",
#                  default_dirpath=os.path.expanduser("~"),
#                  extension_filter=""):
#         """
#         """
#         super().__init__(parent)
#         #
#         self.dirpath = default_dirpath
#         self.filter = extension_filter
#         #
#         self.main_layout = QtWidgets.QHBoxLayout(self)
#         self.path_box = QtWidgets.QLineEdit(default)
#         self.dialog_b = QtWidgets.QPushButton(self.BUTTON_TXT)
#         #
#         self.main_layout.addWidget(self.path_box)
#         self.main_layout.addWidget(self.dialog_b)
#         #
#         self.dialog_b.pressed.connect(self.path_dialog)

#     def path_dialog(self):
#         """
#         """
#         filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
#             parent=self, dir=self.dirpath, filter=self.filter)
#         if filepath:
#             self.dirpath = os.path.dirname(filepath)
#             self.path_box.clear()
#             self.path_box.insert(filepath)

#     def get_state(self):
#         """
#         """
#         return self.path_box.text()


# # #############################################################################
# # # COMPOSITE WIDGETS
# # #############################################################################
# class FileList(QtWidgets.QWidget):
#     """
#     A file dialog button followed by a list that shows the files in the
#     selected folder.
#     """
#     def __init__(self, label, parent=None,
#                  default_path=None, extensions=None, sort=True):
#         """
#         :param extensions: A list of string terminations to match, or ``None``
#           for match-all.
#         :param default_path: If None, 'home' is picked as default.
#         :param sort: If true, contents will always be shown sorted
#         """
#         super().__init__(parent)
#         self.sort = sort
#         self.dirpath = (os.path.expanduser("~") if default_path is None
#                         else default_path)
#         self.extensions = [""] if extensions is None else extensions
#         self.label = label
#         # create widgets
#         self.file_button = QtWidgets.QPushButton(self.label)
#         self.file_list = QtWidgets.QListWidget()
#         # add widgets to layout
#         self.main_layout = QtWidgets.QVBoxLayout(self)
#         self.main_layout.addWidget(self.file_button)
#         self.main_layout.addWidget(self.file_list)
#         # connect
#         self.file_button.pressed.connect(self._file_dialog_handler)
#         # track filesystem changes
#         self.file_watcher = QtCore.QFileSystemWatcher()
#         self.file_watcher.fileChanged.connect(
#             lambda: self.update_path(self.dirpath))
#         self.file_watcher.directoryChanged.connect(
#             lambda: self.update_path(self.dirpath))

#     def update_path(self, dirname):
#         """
#         :param str dirname: The new directory path to be listed.
#         """
#         file_names = [f for f in os.listdir(dirname)
#                       if f.lower().endswith(tuple(self.extensions))]
#         self.file_list.clear()
#         self.file_list.addItems(file_names)
#         #
#         self.file_watcher.removePath(self.dirpath)
#         self.file_watcher.addPath(dirname)
#         #
#         self.dirpath = dirname
#         if self.sort:
#             self.file_list.sortItems(QtCore.Qt.AscendingOrder)

#     def _file_dialog_handler(self):
#         """
#         Opens a file dialog which returns the selected path.
#         """
#         dirname = QtWidgets.QFileDialog.getExistingDirectory(
#             self, self.label, self.dirpath)
#         if dirname:
#             self.update_path(dirname)


# class ScrollbarList(QtWidgets.QWidget):
#     """
#     Override the ``setup_adder_layout(lyt)`` method with your preferred GUI
#     for adding elements. Then overrider ``add_element(*args, **kwargs)`` to
#     add elements accordingly to ``self.list_layout``.
#     """
#     def __init__(self, parent, horizontal=False):
#         """
#         """

#         super().__init__(parent)
#         self.horizontal = horizontal
#         # Structure: main_layout->scroller_widget->dummy_widget->inner_layout
#         # and inner_layout holds the adder_layout and the list_layout
#         self.main_layout = (QtWidgets.QHBoxLayout(self) if horizontal
#                             else QtWidgets.QVBoxLayout(self))
#         self.inner_layout = (QtWidgets.QHBoxLayout(self) if horizontal
#                             else QtWidgets.QVBoxLayout())
#         # The adder layout usually holds an "add" button, but it can also
#         # hold more complex adding mechanisms.
#         self.adder_layout = (QtWidgets.QHBoxLayout() if not horizontal
#                             else QtWidgets.QVBoxLayout())
#         self.setup_adder_layout(self.adder_layout)
#         # the list layout will be dynamically modified by the user
#         self.list_layout = (QtWidgets.QHBoxLayout() if horizontal
#                             else QtWidgets.QVBoxLayout())
#         # inner layout holds the adder and the list layout
#         # self.inner_layout.addLayout(self.adder_layout)
#         self.inner_layout.addLayout(self.list_layout)
#         # create and populate scroll area
#         scroll_bars = (False, True) if horizontal else (True, False)
#         scroller, scroller_widget = get_scroll_area(*scroll_bars)
#         scroller_widget.setLayout(self.inner_layout)
#         # finally add scroll area to main layout
#         self.main_layout.addLayout(self.adder_layout)
#         self.main_layout.addWidget(scroller)

#     def setup_adder_layout(self, lyt):
#         """
#         """
#         add_elt_b = QtWidgets.QPushButton("Add Element")
#         lyt.addWidget(add_elt_b)
#         add_elt_b.pressed.connect(self.add_element)

#     def add_element(self):
#         """
#         """
#         # Create and add a button
#         dummy_button = QtWidgets.QPushButton("Delete Me!")
#         self.list_layout.addWidget(dummy_button)
#         # When pressed, the button deletes itself from the list
#         def delete_me():
#             b_idx = self.list_layout.indexOf(dummy_button)
#             b = self.list_layout.takeAt(b_idx)
#             recursive_delete_qt(b)
#         dummy_button.pressed.connect(delete_me)


# class CheckBoxGroup(QtWidgets.QWidget):
#     """
#     A group of ``CheckBox`` es
#     """
#     def __init__(self, parent=None, horizontal=False):
#         """
#         """
#         super().__init__(parent=parent)
#         ly = QtWidgets.QHBoxLayout() if horizontal else QtWidgets.QVBoxLayout()
#         self.setLayout(ly)

#     def add_box(self, name, tristate=False, initial_val=True):
#         """
#         :param bool tristate: If true, the added check box will have 3 states.
#         """
#         b = QtWidgets.QCheckBox(name)
#         b.setTristate(tristate)
#         b.setChecked(initial_val)
#         self.layout().addWidget(b)

#     def remove_box(self, idx):
#         """
#         :param int idx: Boxes are added in increasing index order, so this
#           the lower this index the 'older' the box that is being removed.
#         """
#         b = self.layout().takeAt(idx)
#         b.widget().setParent(QtWidgets.QWidget())  # this is needed...

#     def state(self):
#         """
#         :returns: A list with all the current states, in index order.
#         """
#         ly = self.layout()
#         return [ly.itemAt(i).widget().checkState() for i in range(ly.count())]


# # #############################################################################
# # # FORMS
# # #############################################################################
# class SaveForm(QtWidgets.QWidget):
#     """
#     A formulary providing functionality for selecting what to save, where to
#     save, the output suffix and overwriting policy.
#     """

#     SAVE_TEXT = "Save\nselected"
#     DIALOG_TEXT = "Output\nfolder"
#     OVERWRITE_TEXT = "Overwrite\nsaved"

#     def __init__(self, parent=None, default_path=None):
#         """
#         :param str default_path: If not given, 'home' is picked as default.
#         """
#         super().__init__(parent)
#         self.save_path = (os.path.expanduser("~") if default_path is None
#                           else default_path)
#         # create widgets
#         self.save_group = CheckBoxGroup()
#         self.text_boxes = QtWidgets.QVBoxLayout()
#         self.file_dialog_button = QtWidgets.QPushButton(self.DIALOG_TEXT)
#         self.save_button = QtWidgets.QPushButton(self.SAVE_TEXT)
#         self.overwrite_button = QtWidgets.QCheckBox(self.OVERWRITE_TEXT)
#         self.overwrite_button.setChecked(True)
#         # build layout hierarchy
#         self.main_layout = QtWidgets.QHBoxLayout()
#         self.main_layout.addWidget(self.save_group)
#         self.main_layout.addLayout(self.text_boxes)
#         #
#         lyt = QtWidgets.QVBoxLayout()
#         lyt.addWidget(self.file_dialog_button)
#         lyt.addWidget(self.save_button)
#         lyt.addWidget(self.overwrite_button)
#         self.main_layout.addLayout(lyt)
#         self.setLayout(self.main_layout)
#         # add connections
#         self.file_dialog_button.clicked.connect(self._change_save_path)
#         self.save_button.clicked.connect(self._handle_save_masks)

#     def add_checkbox(self, checkbox_name, initial_val=True, initial_txt=None):
#         """
#         Adds an element that can be selected to be saved.

#         :param checkbox_name: The element identifier
#         :param initial_txt: The initial suffix to be appended to the files. If
#           none is given, the ``checkbox_name`` is picked as default. The user
#           can change this from the GUI.
#         """
#         if initial_txt is None:
#             initial_txt = checkbox_name
#         tb = QtWidgets.QLineEdit(initial_txt)
#         #
#         self.save_group.add_box(checkbox_name, False, initial_val)
#         self.text_boxes.addWidget(tb)

#     def _change_save_path(self):
#         """
#         """
#         dir_name = QtWidgets.QFileDialog.getExistingDirectory(
#             self, self.DIALOG_TEXT, self.save_path)
#         if dir_name:
#             self.save_path = dir_name

#     def _handle_save_masks(self):
#         """
#         """
#         states = [s == QtCore.Qt.CheckState.Checked
#                   for s in self.save_group.state()]
#         suffixes = [self.text_boxes.itemAt(i).widget().text()
#                     for i in range(self.text_boxes.count())]
#         overwrite = self.overwrite_button.isChecked()
#         self.save_masks(states, suffixes, overwrite)

#     def save_masks(self, states, suffixes, overwrite):
#         """
#         :param states: A list with booleans, representing the checkbox
#           states for the contained elements.
#         :param suffixes: A list with the corresponding suffixes
#         :param overwrite: A boolean determining whether the 'overwrite'
#           checkbox has been activated.

#         Override me!
#         """
#         print("check boxes:", states, "suffixes:", suffixes,
#               "out path:", self.save_path, "overwrite:", overwrite)


class NamedForm(QtWidgets.QWidget):
    """
    This convenience class allows to create a QFormLayout with a title on top.
    It also features a method to retrieve the current state as a dictionary.

    :cvar int TITLE_WEIGHT: From 0 to 100, 50 is normal, >50 is bold font
    """
    TITLE_WEIGHT = 63

    def __init__(self, form_param_nwd, form_name=None, parent=None):
        """
        :param form_name: A title to be displayed at the top center with
          ``self.TITLE_WEIGHT`` from 0 to 100 (50 is normal, more is bold).
        :param form_param_nwd: A list of parameters in the form
          ``(name, widget_fn, default)``. The widget_fn must accept a single
          ``default`` input, and return the Qt object (e.g. a widget).
          the signature: ``widget = widget_type(self, default)``. Furthermore,
          the returned widget must implement a ``get_state()`` method that
          returns its current state (e.g. a checkbox returns True or False).
        """
        super().__init__(parent)
        #
        self.name = form_name
        self.form_param_nwd = form_param_nwd
        # create main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)
        # create and configure name label. put label inside its own layout
        # so we may add widgets on the same level, like "select" buttons
        if form_name is not None:
            self.title_layout = QtWidgets.QHBoxLayout()
            self.title = QtWidgets.QLabel(form_name)
            self.title.setAlignment(QtCore.Qt.AlignCenter)
            change_label_font(self.title, weight=self.TITLE_WEIGHT)
            self.title_layout.addWidget(self.title)
            # add to main layout
            self.main_layout.addLayout(self.title_layout)
        # create named form and add to main layout
        self.form = self.get_parameter_form(form_param_nwd)
        self.main_layout.addLayout(self.form)

    @classmethod
    def get_parameter_form(cls, nwd_list):
        """
        :param nwd_list: See explanation in constructor.

        This method creates and returns a form layout with the given parameter
        name, data type and default.
        """
        form_layout = QtWidgets.QFormLayout()
        for pname, w_fn, pdefault in nwd_list:
            qobj = w_fn(default=pdefault)
            form_layout.addRow(pname, qobj)
        return form_layout

    def get_state(self):
        """
        :returns: A dict in the form ``param_name: param_value``.
        """
        result = {}
        for i in range(self.form.rowCount()):
            par_name = self.form.itemAt(i, self.form.LabelRole).widget().text()
            # Note that we expect w to implement a get_state() method that
            # returns its current state
            w = self.form.itemAt(i, self.form.FieldRole).widget()
            par_val = w.get_state()
            result[par_name] = par_val
        #
        return result

    def get_widgets(self):
        """
        """
        result = {}
        for i in range(self.form.rowCount()):
            par_name = self.form.itemAt(i, self.form.LabelRole).widget().text()
            w = self.form.itemAt(i, self.form.FieldRole).widget()
            result[par_name] = w
        #
        return result
