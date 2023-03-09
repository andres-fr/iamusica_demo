# -*- coding:utf-8 -*-


"""
This module is a library of reusable, extendable Qt widgets.
"""


from PySide2 import QtWidgets, QtCore
#
from .. import change_label_font


# ##############################################################################
# # BASIC WIDGETS
# ##############################################################################
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
        super().__init__(*args, **kargs)
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
