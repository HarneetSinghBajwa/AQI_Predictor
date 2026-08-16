import sys
import math
import joblib

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QDoubleValidator,
    QPainter,
    QPainterPath,
    QColor,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_FILE = "AQI_model.pkl"

FEATURES = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "Temperature",
    "Humidity",
]


# ==================================================
# AQI CATEGORY
# ==================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "GOOD"

    elif aqi <= 100:
        return "MODERATE"

    elif aqi <= 150:
        return "UNHEALTHY FOR SENSITIVE GROUPS"

    elif aqi <= 200:
        return "UNHEALTHY"

    elif aqi <= 300:
        return "VERY UNHEALTHY"

    else:
        return "HAZARDOUS"


# ==================================================
# CORNER LEAVES
# ==================================================

class CornerLeaves(QWidget):

    def __init__(self, corner, parent=None):

        super().__init__(parent)

        self.corner = corner

        self.setAttribute(
            Qt.WA_TransparentForMouseEvents
        )

        self.setAttribute(
            Qt.WA_NoSystemBackground
        )

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.Antialiasing
        )

        leaf_color = QColor("#C2B15D")
        leaf_edge = QColor("#9D8D42")
        stem_color = QColor("#9B8E4A")

        # --------------------------------------------------
        # Draw origami leaf
        # --------------------------------------------------

        def draw_leaf(x, y, angle, size):

            painter.save()

            painter.translate(x, y)
            painter.rotate(angle)

            path = QPainterPath()

            path.moveTo(0, 0)

            path.quadTo(
                size * 0.25,
                -size * 0.65,
                size,
                -size * 0.35
            )

            path.quadTo(
                size * 0.62,
                size * 0.02,
                0,
                0
            )

            painter.setBrush(
                leaf_color
            )

            painter.setPen(
                QPen(
                    leaf_edge,
                    1
                )
            )

            painter.drawPath(path)

            # Fold line
            painter.setPen(
                QPen(
                    QColor("#8E8040"),
                    1
                )
            )

            painter.drawLine(
                QPointF(0, 0),
                QPointF(
                    size * 0.78,
                    -size * 0.30
                )
            )

            painter.restore()

        # --------------------------------------------------
        # Draw branch
        # --------------------------------------------------

        def draw_branch(points):

            painter.setPen(
                QPen(
                    stem_color,
                    2
                )
            )

            for i in range(len(points) - 1):

                painter.drawLine(
                    QPointF(*points[i]),
                    QPointF(*points[i + 1])
                )

        w = self.width()
        h = self.height()

        # ==================================================
        # TOP LEFT
        # ==================================================

        if self.corner == "top_left":

            draw_branch([
                (8, 112),
                (35, 83),
                (63, 58),
                (92, 37),
                (124, 18),
            ])

            draw_leaf(18, 100, -40, 26)
            draw_leaf(43, 77, 18, 25)
            draw_leaf(68, 55, -24, 28)
            draw_leaf(94, 37, 15, 26)
            draw_leaf(122, 20, -12, 27)

        # ==================================================
        # TOP RIGHT
        #
        # IMPORTANT:
        # The entire branch is kept in a small strip at
        # the extreme upper-right corner.
        #
        # It never reaches the vertical position of
        # "Model ready".
        # ==================================================

        elif self.corner == "top_right":

            draw_branch([
                (w - 1, 3),
                (w - 15, 9),
                (w - 30, 17),
                (w - 46, 26),
                (w - 63, 37),
            ])

            draw_leaf(
                w - 4,
                4,
                205,
                16
            )

            draw_leaf(
                w - 18,
                10,
                178,
                15
            )

            draw_leaf(
                w - 33,
                18,
                202,
                16
            )

            draw_leaf(
                w - 49,
                27,
                180,
                15
            )

            draw_leaf(
                w - 65,
                38,
                202,
                16
            )

        # ==================================================
        # BOTTOM LEFT
        # ==================================================

        elif self.corner == "bottom_left":

            draw_branch([
                (8, h - 8),
                (42, h - 27),
                (76, h - 48),
                (110, h - 71),
                (146, h - 94),
            ])

            draw_leaf(
                18,
                h - 13,
                -18,
                26
            )

            draw_leaf(
                49,
                h - 30,
                15,
                25
            )

            draw_leaf(
                81,
                h - 50,
                -15,
                27
            )

            draw_leaf(
                114,
                h - 72,
                13,
                26
            )

            draw_leaf(
                143,
                h - 93,
                -12,
                27
            )

        # ==================================================
        # BOTTOM RIGHT
        # ==================================================

        elif self.corner == "bottom_right":

            draw_branch([
                (w - 8, h - 8),
                (w - 42, h - 27),
                (w - 76, h - 48),
                (w - 110, h - 71),
                (w - 146, h - 94),
            ])

            draw_leaf(
                w - 18,
                h - 13,
                198,
                26
            )

            draw_leaf(
                w - 49,
                h - 30,
                165,
                25
            )

            draw_leaf(
                w - 81,
                h - 50,
                195,
                27
            )

            draw_leaf(
                w - 114,
                h - 72,
                163,
                26
            )

            draw_leaf(
                w - 143,
                h - 93,
                192,
                27
            )

        painter.end()


# ==================================================
# MAIN WINDOW
# ==================================================

class AQIPredictor(QWidget):

    def __init__(self):

        super().__init__()

        self.model = None
        self.input_fields = {}

        self.load_model()
        self.setup_window()
        self.setup_ui()

    # ==================================================
    # LOAD MODEL
    # ==================================================

    def load_model(self):

        try:

            self.model = joblib.load(
                MODEL_FILE
            )

        except FileNotFoundError:

            QMessageBox.critical(
                self,
                "Model Error",
                f"'{MODEL_FILE}' was not found.\n\n"
                "Make sure AQI_model.pkl is in the "
                "same folder as GUI.py."
            )

            sys.exit(1)

        except Exception as error:

            QMessageBox.critical(
                self,
                "Model Error",
                f"Unable to load the model.\n\n"
                f"{error}"
            )

            sys.exit(1)

    # ==================================================
    # WINDOW SETUP
    # ==================================================

    def setup_window(self):

        self.setWindowTitle(
            "AQI Predictor"
        )

        self.resize(
            1180,
            760
        )

        self.setMinimumSize(
            950,
            650
        )

        self.setStyleSheet("""

            QWidget {
                background-color: #FBF8F2;
                color: #252322;
                font-family: "Georgia";
            }

            QFrame#inputCard {
                background-color: #FFFFFF;
                border: 1px solid #E8E1D5;
                border-radius: 24px;
            }

            QFrame#resultCard {
                background-color: #F3EEE4;
                border: 1px solid #E4DBCC;
                border-radius: 24px;
            }

            QLabel#title {
                background: transparent;
                color: #262423;
                font-family: "Georgia";
                font-size: 39px;
                font-weight: 700;
            }

            QLabel#subtitle {
                background: transparent;
                color: #756C63;
                font-family: "Georgia";
                font-size: 15px;
            }

            QLabel#sectionTitle {
                background: transparent;
                color: #292624;
                font-family: "Georgia";
                font-size: 30px;
                font-weight: 700;
            }

            QLabel#hint {
                background: transparent;
                color: #8B837A;
                font-family: "Georgia";
                font-size: 13px;
            }

            QLabel#fieldLabel {
                background: transparent;
                color: #514A44;
                font-family: "Georgia";
                font-size: 13px;
                font-weight: 600;
            }

            QLineEdit {
                background-color: #FCFBF8;
                border: 1px solid #DDD5C9;
                border-radius: 14px;
                padding: 12px 14px;
                color: #282522;
                font-family: "Georgia";
                font-size: 14px;
                selection-background-color: #CBD4C3;
            }

            QLineEdit:hover {
                border: 1px solid #C8C0B4;
            }

            QLineEdit:focus {
                border: 1px solid #8D9678;
                background-color: #FFFFFF;
            }

            QPushButton#predictButton {
                background-color: #4E7659;
                color: #FFFFFF;
                border: none;
                border-radius: 14px;
                padding: 14px 24px;
                font-family: "Georgia";
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton#predictButton:hover {
                background-color: #41654B;
            }

            QPushButton#predictButton:pressed {
                background-color: #395D43;
            }

            QPushButton#resetButton {
                background-color: #F4F0E9;
                color: #514A44;
                border: 1px solid #D9D0C4;
                border-radius: 14px;
                padding: 14px 24px;
                font-family: "Georgia";
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton#resetButton:hover {
                background-color: #ECE6DC;
            }

            QLabel#resultCaption {
                background: transparent;
                color: #8A8177;
                font-family: "Georgia";
                font-size: 13px;
            }

            QLabel#resultValue {
                background-color: #FFFDF9;
                color: #292624;
                border: 1px solid #E5DDD1;
                border-radius: 20px;
                padding: 14px 18px;
                font-family: "Georgia";
                font-size: 60px;
                font-weight: 700;
            }

            QLabel#resultCategory {
                background-color: #E5EEE4;
                color: #4E7659;
                border-radius: 12px;
                padding: 8px 12px;
                font-family: "Georgia";
                font-size: 14px;
                font-weight: 700;
            }

            QLabel#resultDescription {
                background-color: #F9F6F0;
                color: #756D65;
                border-radius: 12px;
                padding: 10px 12px;
                font-family: "Georgia";
                font-size: 13px;
            }

            QLabel#modelInfo {
                background-color: #F9F6F0;
                color: #817870;
                border-radius: 12px;
                padding: 12px;
                font-family: "Georgia";
                font-size: 12px;
            }

            QLabel#statusText {
                background: transparent;
                color: #6F675F;
                font-family: "Segoe UI";
                font-size: 12px;
            }

            QLabel#footer {
                background: transparent;
                color: #948A81;
                font-family: "Segoe UI";
                font-size: 11px;
            }

        """)

    # ==================================================
    # BUILD UI
    # ==================================================

    def setup_ui(self):

        main_layout = QVBoxLayout(
            self
        )

        main_layout.setContentsMargins(
            52,
            36,
            52,
            30
        )

        main_layout.setSpacing(
            20
        )

        # ==================================================
        # HEADER
        # ==================================================

        header = QHBoxLayout()

        header.setContentsMargins(
            105,
            0,
            0,
            0
        )

        title_block = QVBoxLayout()

        title_block.setSpacing(
            4
        )

        title = QLabel(
            "AQI Predictor"
        )

        title.setObjectName(
            "title"
        )

        subtitle = QLabel(
            "A simple environmental quality assessment tool"
        )

        subtitle.setObjectName(
            "subtitle"
        )

        title_block.addWidget(
            title
        )

        title_block.addWidget(
            subtitle
        )

        # Model ready remains untouched
        status_block = QHBoxLayout()

        status_block.setSpacing(
            6
        )

        status_dot = QLabel(
            "●"
        )

        status_dot.setStyleSheet(
            """
            QLabel {
                background: transparent;
                color: #6E9A77;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            """
        )

        status_text = QLabel(
            "Model ready"
        )

        status_text.setObjectName(
            "statusText"
        )

        status_block.addWidget(
            status_dot
        )

        status_block.addWidget(
            status_text
        )

        header.addLayout(
            title_block
        )

        header.addStretch()

        header.addLayout(
            status_block
        )

        main_layout.addLayout(
            header
        )

        # ==================================================
        # MAIN CONTENT
        # ==================================================

        content = QHBoxLayout()

        content.setSpacing(
            24
        )

        # ==================================================
        # INPUT CARD
        # ==================================================

        input_card = QFrame()

        input_card.setObjectName(
            "inputCard"
        )

        input_layout = QVBoxLayout(
            input_card
        )

        input_layout.setContentsMargins(
            28,
            28,
            28,
            28
        )

        # --------------------------------------------------
        # Heading and hint
        # --------------------------------------------------

        section_block = QVBoxLayout()

        section_block.setSpacing(
            0
        )

        section_title = QLabel(
            "Environmental inputs"
        )

        section_title.setObjectName(
            "sectionTitle"
        )

        hint = QLabel(
            "Enter the values used by the regression model."
        )

        hint.setObjectName(
            "hint"
        )

        section_block.addWidget(
            section_title
        )

        section_block.addWidget(
            hint
        )

        input_layout.addLayout(
            section_block
        )

        input_layout.addSpacing(
            22
        )

        # ==================================================
        # INPUT GRID
        # ==================================================

        grid = QGridLayout()

        grid.setHorizontalSpacing(
            18
        )

        grid.setVerticalSpacing(
            15
        )

        for index, feature in enumerate(
            FEATURES
        ):

            row = index // 2
            column = index % 2

            field_layout = QVBoxLayout()

            field_layout.setSpacing(
                6
            )

            label = QLabel(
                self.display_name(
                    feature
                )
            )

            label.setObjectName(
                "fieldLabel"
            )

            field = QLineEdit()

            field.setPlaceholderText(
                "Enter value"
            )

            field.setValidator(
                QDoubleValidator(
                    -999999.0,
                    999999.0,
                    4,
                    field
                )
            )

            field.returnPressed.connect(
                self.predict
            )

            field_layout.addWidget(
                label
            )

            field_layout.addWidget(
                field
            )

            grid.addLayout(
                field_layout,
                row,
                column
            )

            self.input_fields[
                feature
            ] = field

        input_layout.addLayout(
            grid
        )

        # --------------------------------------------------
        # Slightly more space before buttons
        # --------------------------------------------------

        input_layout.addSpacing(
            18
        )

        buttons = QHBoxLayout()

        buttons.setSpacing(
            10
        )

        predict_button = QPushButton(
            "Predict AQI"
        )

        predict_button.setObjectName(
            "predictButton"
        )

        predict_button.clicked.connect(
            self.predict
        )

        reset_button = QPushButton(
            "Reset"
        )

        reset_button.setObjectName(
            "resetButton"
        )

        reset_button.clicked.connect(
            self.reset_fields
        )

        buttons.addWidget(
            predict_button,
            1
        )

        buttons.addWidget(
            reset_button,
            1
        )

        input_layout.addLayout(
            buttons
        )

        content.addWidget(
            input_card,
            3
        )

        # ==================================================
        # RESULT CARD
        # ==================================================

        result_card = QFrame()

        result_card.setObjectName(
            "resultCard"
        )

        result_layout = QVBoxLayout(
            result_card
        )

        result_layout.setContentsMargins(
            34,
            34,
            34,
            34
        )

        result_layout.setSpacing(
            12
        )

        result_caption = QLabel(
            "PREDICTED AIR QUALITY INDEX"
        )

        result_caption.setObjectName(
            "resultCaption"
        )

        self.result_value = QLabel(
            "--"
        )

        self.result_value.setObjectName(
            "resultValue"
        )

        self.result_value.setAlignment(
            Qt.AlignLeft |
            Qt.AlignVCenter
        )

        self.result_category = QLabel(
            "WAITING FOR INPUT"
        )

        self.result_category.setObjectName(
            "resultCategory"
        )

        self.result_description = QLabel(
            "Enter the environmental measurements "
            "to generate an AQI estimate."
        )

        self.result_description.setObjectName(
            "resultDescription"
        )

        self.result_description.setWordWrap(
            True
        )

        divider = QFrame()

        divider.setFrameShape(
            QFrame.HLine
        )

        divider.setStyleSheet(
            """
            QFrame {
                background-color: #DED5C8;
                border: none;
                min-height: 1px;
                max-height: 1px;
            }
            """
        )

        model_info = QLabel(
            "Model\n"
            "Multiple Linear Regression\n\n"
            "Inputs\n"
            "8 environmental parameters"
        )

        model_info.setObjectName(
            "modelInfo"
        )

        result_layout.addWidget(
            result_caption
        )

        result_layout.addSpacing(
            6
        )

        result_layout.addWidget(
            self.result_value
        )

        result_layout.addWidget(
            self.result_category
        )

        result_layout.addWidget(
            self.result_description
        )

        result_layout.addSpacing(
            8
        )

        result_layout.addWidget(
            divider
        )

        result_layout.addSpacing(
            8
        )

        result_layout.addWidget(
            model_info
        )

        result_layout.addStretch()

        content.addWidget(
            result_card,
            2
        )

        main_layout.addLayout(
            content,
            1
        )

        # ==================================================
        # FOOTER
        # ==================================================

        footer = QLabel(
            "AQI Prediction  •  Supervised Regression"
        )

        footer.setObjectName(
            "footer"
        )

        footer.setAlignment(
            Qt.AlignCenter
        )

        main_layout.addWidget(
            footer
        )

        # ==================================================
        # FOUR CORNER LEAVES
        # ==================================================

        self.top_left_leaves = CornerLeaves(
            "top_left",
            self
        )

        self.top_right_leaves = CornerLeaves(
            "top_right",
            self
        )

        self.bottom_left_leaves = CornerLeaves(
            "bottom_left",
            self
        )

        self.bottom_right_leaves = CornerLeaves(
            "bottom_right",
            self
        )

        self.update_leaf_positions()

        self.top_left_leaves.lower()
        self.top_right_leaves.lower()
        self.bottom_left_leaves.lower()
        self.bottom_right_leaves.lower()

    # ==================================================
    # POSITION LEAVES
    # ==================================================

    def update_leaf_positions(self):

        # ----------------------------------------------
        # Top-left
        # ----------------------------------------------

        self.top_left_leaves.setGeometry(
            8,
            28,
            155,
            115
        )

        # ----------------------------------------------
        # Top-right
        #
        # This is intentionally tiny and very high.
        # Its lowest point stays above Model ready.
        # ----------------------------------------------

        self.top_right_leaves.setGeometry(
            self.width() - 94,
            12,
            88,
            56
        )

        # ----------------------------------------------
        # Bottom-left
        # ----------------------------------------------

        self.bottom_left_leaves.setGeometry(
            8,
            self.height() - 112,
            190,
            104
        )

        # ----------------------------------------------
        # Bottom-right
        # ----------------------------------------------

        self.bottom_right_leaves.setGeometry(
            self.width() - 198,
            self.height() - 112,
            190,
            104
        )

    # ==================================================
    # RESPONSIVE LEAVES
    # ==================================================

    def resizeEvent(self, event):

        super().resizeEvent(
            event
        )

        self.update_leaf_positions()

    # ==================================================
    # DISPLAY NAMES
    # ==================================================

    def display_name(
        self,
        feature
    ):

        names = {

            "PM2.5":
                "PM2.5  ·  µg/m³",

            "PM10":
                "PM10  ·  µg/m³",

            "NO2":
                "NO₂  ·  ppb",

            "SO2":
                "SO₂  ·  ppb",

            "CO":
                "CO  ·  ppm",

            "O3":
                "O₃  ·  ppb",

            "Temperature":
                "Temperature  ·  °C",

            "Humidity":
                "Humidity  ·  %",

        }

        return names[
            feature
        ]

    # ==================================================
    # VALIDATE INPUTS
    # ==================================================

    def validate_inputs(self):

        values = []

        for feature in FEATURES:

            text = (
                self.input_fields[
                    feature
                ]
                .text()
                .strip()
            )

            if not text:

                raise ValueError(
                    f"Please enter "
                    f"{self.display_name(feature)}."
                )

            try:

                value = float(
                    text
                )

            except ValueError:

                raise ValueError(
                    f"{self.display_name(feature)} "
                    f"must be numeric."
                )

            if not math.isfinite(
                value
            ):

                raise ValueError(
                    f"{self.display_name(feature)} "
                    f"contains an invalid value."
                )

            if (
                feature != "Temperature"
                and value < 0
            ):

                raise ValueError(
                    f"{self.display_name(feature)} "
                    f"cannot be negative."
                )

            if feature == "Humidity":

                if (
                    value < 0
                    or value > 100
                ):

                    raise ValueError(
                        "Humidity must be "
                        "between 0 and 100%."
                    )

            values.append(
                value
            )

        return values

    # ==================================================
    # PREDICT
    # ==================================================

    def predict(self):

        try:

            values = (
                self.validate_inputs()
            )

        except ValueError as error:

            QMessageBox.warning(
                self,
                "Invalid Input",
                str(error)
            )

            return

        try:

            prediction = (
                self.model.predict(
                    [values]
                )[0]
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Prediction Error",
                "Unable to generate prediction.\n\n"
                f"{error}"
            )

            return

        if not math.isfinite(
            prediction
        ):

            QMessageBox.critical(
                self,
                "Prediction Error",
                "The model returned "
                "an invalid prediction."
            )

            return

        prediction = max(
            0.0,
            prediction
        )

        category = get_aqi_category(
            prediction
        )

        self.result_value.setText(
            f"{prediction:.1f}"
        )

        self.result_category.setText(
            category
        )

        self.result_description.setText(
            "AQI estimated from PM2.5, PM10, "
            "NO₂, SO₂, CO, O₃, temperature "
            "and humidity."
        )

    # ==================================================
    # RESET
    # ==================================================

    def reset_fields(self):

        for field in (
            self.input_fields.values()
        ):

            field.clear()

        self.result_value.setText(
            "--"
        )

        self.result_category.setText(
            "WAITING FOR INPUT"
        )

        self.result_description.setText(
            "Enter the environmental measurements "
            "to generate an AQI estimate."
        )


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

def main():

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "AQI Predictor"
    )

    window = AQIPredictor()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()