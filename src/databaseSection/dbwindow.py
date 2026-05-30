import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
import csv 




#  SIDEBAR NAV BUTTON

class NavButton(QPushButton):

    def __init__(self, label, parent=None):
        super().__init__(label, parent)
        self.setCheckable(True)
        self.setFixedHeight(35)


#  STAT CARD  (reusable placeholder)

class StatCard(QFrame):
    """Top-row KPI card: title + big number + subtitle."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        self.title_lbl = QLabel(title)
        self.value_lbl = QLabel("—")
        self.sub_lbl   = QLabel("")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.sub_lbl)

    def set_value(self, value:int, subtitle=""):
        if self.title_lbl.text() != "Alerts" :
            self.value_lbl.setText(f"{str(value)} %")
        else : self.value_lbl.setText(str(value))
        self.sub_lbl.setText(subtitle)



#  CHART PLACEHOLDER

class ChartPlaceholder(QFrame):
    """Blank frame that will hold a QtChart later."""
    def __init__(self, label="Chart", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)



#  PANEL 1 — DASHBOARD

class DashboardPanel(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(16)

        # ── Header row ──────────────────────────────────
        header = QHBoxLayout()
        self.page_title = QLabel("Dashboard")
        self.page_title.setProperty("title", True)
        self.session_badge = QLabel("No active session")
        self.export_btn = QPushButton("Export Report")
        header.addWidget(self.page_title)
        header.addStretch()
        header.addWidget(self.session_badge)
        header.addWidget(self.export_btn)
        main_w.addLayout(header)

        # ── Stat cards row ───────────────────────────────
        cards_row = QHBoxLayout()
        self.card_present  = StatCard("Present")
        self.card_present.set_value(80)
        self.card_late     = StatCard("Late")
        self.card_late.set_value(0)
        self.card_absent   = StatCard("Absent")
        self.card_absent.set_value(20)
        self.card_alerts   = StatCard("Alerts")
        self.card_alerts.set_value(20)
        for card in [self.card_present, self.card_late,
                     self.card_absent, self.card_alerts]:
            cards_row.addWidget(card)
        main_w.addLayout(cards_row)

        
        # ── Bottom row: by-year box + recent log ───────
       

        self.bot_row_lt = QHBoxLayout()
        self.bot_row_lt.setSpacing(16)

        self.att_by_year_box = QGroupBox("Attendance by year")
        self.att_by_year_lt = QHBoxLayout(self.att_by_year_box)

        for year in ["1st year ", "2nd year", "3rd year"]:
            self.att_by_year_lt.addWidget(ChartPlaceholder(year))

        self.att_by_year_box.setFixedWidth(400)
        self.att_by_year_box.setFixedHeight(200)

        # Recent access log mini-table
        log_frame = QFrame()
        log_frame.setMaximumHeight(350)
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Recent activity"))
        log_header.addStretch()
        self.view_all_btn = QPushButton("View all")
        log_header.addWidget(self.view_all_btn)
        log_layout.addLayout(log_header)

        self.recent_log_table = QTableWidget(0, 3)
        self.recent_log_table.setHorizontalHeaderLabels(
            ["Student", "Time", "Event"])
        # testing rows 
        row = self.recent_log_table.rowCount()
        self.recent_log_table.insertRow(row)

        self.recent_log_table.setItem(row, 0, QTableWidgetItem("Ahmed kanabawi"))
        self.recent_log_table.setItem(row, 1, QTableWidgetItem("08:15:29"))
        self.recent_log_table.setItem(row, 2, QTableWidgetItem("Entry"))

        row +=1
        self.recent_log_table.insertRow(row)

        self.recent_log_table.setItem(row, 0, QTableWidgetItem("Mohammed sonbol"))
        self.recent_log_table.setItem(row, 1, QTableWidgetItem("22:19:29"))
        self.recent_log_table.setItem(row, 2, QTableWidgetItem("Exit"))
       
       

        self.recent_log_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.recent_log_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recent_log_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.recent_log_table.verticalHeader().setVisible(False)
        log_layout.addWidget(self.recent_log_table)

        self.bot_row_lt.addWidget(self.att_by_year_box)
        self.bot_row_lt.addWidget(log_frame)
        main_w.addSpacing(50)
        main_w.addLayout(self.bot_row_lt)

    def on_enter(self):
        """Called when this panel becomes visible — refresh data here.  """
        pass


# ═════════════════════════════════════════════
#  PANEL 2 — STUDENTS TABLE
# ═════════════════════════════════════════════
class StudentsPanel(QWidget):
    
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()
     
        
    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(12)

        # ── Header ───────────────────────────────────────
        header = QHBoxLayout()
        self.page_title = QLabel("Students")
        self.total_label = QLabel("0 students")
        header.addWidget(self.page_title)
        header.addStretch()
        header.addWidget(self.total_label)
        main_w.addLayout(header)

        # ── Toolbar: search + filters ─────────────────────
        toolbar = QHBoxLayout()
        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("Search by name, ID, or section...")
        self.search_field.setFixedWidth(300)

        

        self.year_filter = QComboBox()
        self.year_filter.addItems(["All years", "1st year", "2nd year","3rd year"])

        toolbar.addWidget(self.search_field)
        toolbar.addWidget(self.year_filter)
        
        toolbar.addStretch()
        main_w.addLayout(toolbar)

        # ── Main table ───────────────────────────────────
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Name", "Lastname", "Section"])
        
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        main_w.addWidget(self.table)

        # ── Footer: action buttons ────────────────────────
        footer = QHBoxLayout()
        self.inspect_btn = QPushButton("Inspect Student")
        self.inspect_btn.setEnabled(False)
        self.delete_btn  = QPushButton("Delete Selected")
        self.delete_btn.setEnabled(False)
        footer.addStretch()
        footer.addWidget(self.inspect_btn)
        footer.addWidget(self.delete_btn)
        main_w.addLayout(footer)

        # Connect selection → enable buttons
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.search_field.textChanged.connect(self.apply_filter)
        self.year_filter.currentIndexChanged.connect(self.apply_filter)
        self.inspect_btn.clicked.connect(self._on_inspect)
        self.delete_btn.clicked.connect(self._on_delete)

    def _on_selection_changed(self):
        has_selection = bool(self.table.selectedItems())        
        self.inspect_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        
    def on_enter(self):
        self.populate(self.db.students)
    def populate(self,students:list):
        self.table.setSortingEnabled(False)
        self.table.setUpdatesEnabled(False)
        self.table.setRowCount(0)
        
      
        for student in students : 
            row = self.table.rowCount()
            self.table.insertRow(row)
        
            for col,info in enumerate(student) : 
                self.table.setItem(row,col,QTableWidgetItem(str(info)))
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)
        self.total_label.setText(f"{self.table.rowCount()} student")
  
    def apply_filter(self):
        query = self.search_field.text().strip().lower()
        year = self.year_filter.currentIndex()

        targets = query.split()

        matches = []

        for student in self.db.students:

            student_year = int(str(student[3])[0])
            if year != 0 and student_year != year:
                continue

            
            if targets:
                if not all(
                    any(target in str(info).lower() for info in student)
                    for target in targets
                ):
                    continue

            matches.append(student)

        self.populate(matches)
            

    def _on_inspect(self):
            selected_items = self.table.selectedItems()
            selected_rows = set(item.row() for item in selected_items)
            if len(selected_rows) > 1 : 
                QMessageBox.warning(self,"Mutliple selection","Please select only one student to inspect")
                return 
            if len(selected_rows) == 0 : 
                return 
            
            # we are left with only one row 
            # currentRow returns the index of the last row that received focus ,
            #  here it is exactly the selected row since there is only one row selected 

            row = selected_rows.pop()
            student= [self.table.item(row, col).text() for col in range(0,4)]
            dlg = StudentInspectDialog(student, self.db, self)
            dlg.exec()

    def _on_delete(self):
        # Auth gate before destructive action
        auth = AuthDialog("delete this student", self)
        if auth.exec() != QDialog.DialogCode.Accepted:
            return
        if auth.get_password() != self.db.DB_PASS :
            QMessageBox.information(self,"Invalid password","Password Invalid please check again")
            return 
        confirm = QMessageBox.question(
            self, "Confirm delete",
            "This will permanently remove the student and all their data.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            pass
            

           

# ─────────────────────────────────────────────
#  STUDENT INSPECT DIALOG
# ─────────────────────────────────────────────
class StudentInspectDialog(QDialog):
    """Shows full details of one student + their attendance history."""
    def __init__(self, student:list[str], db=None, parent=None):
        super().__init__(parent)
        self.student = student
        self.db = db 
        self.embeddingscount = self.db.get_embeddingCount(student[0])
        self.student_history = self.db.get_student_history(student[0])
        self.setWindowTitle("Student Details")
        self.resize(500, 600)
        self._build()
        self.populate_table(self.student_history)
    
    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setSpacing(16)

        # Info section
        info_group = QGroupBox("Personal Info")
        info_layout = QFormLayout(info_group)
        self.name_lbl     = QLabel(self.student[1])
        self.lastname_lbl = QLabel(self.student[2])
        self.section_lbl  = QLabel(self.student[3])
        info_layout.addRow("Name :",        self.name_lbl)
        info_layout.addRow("Lastname :",    self.lastname_lbl)
        info_layout.addRow("Section :",     self.section_lbl)

        main_w.addWidget(info_group)

        # Embeddings info
        emb_group = QGroupBox("Face Embeddings")
        emb_layout = QHBoxLayout(emb_group)
        self.emb_count_lbl = QLabel(f"{self.embeddingscount} vectors stored")
        self.re_register_btn = QPushButton("Re-register Face")
        emb_layout.addWidget(self.emb_count_lbl)
        emb_layout.addStretch()
        emb_layout.addWidget(self.re_register_btn)
        main_w.addWidget(emb_group)

        # Attendance summary for this student
        att_group = QGroupBox("Attendance History")
        att_layout = QVBoxLayout(att_group)
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(["Date", "Exit","Entry"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        att_layout.addWidget(self.history_table)
        main_w.addWidget(att_group)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        main_w.addWidget(self.close_btn, alignment=Qt.AlignRight)
    def populate_table(self,student_history) : 

        self.history_table.setSortingEnabled(False)
        self.history_table.setUpdatesEnabled(False)
        self.history_table.setRowCount(0)

        for event in student_history : 
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
        
            for col,info in enumerate(event) : 
                self.history_table.setItem(row,col,QTableWidgetItem(str(info)))
        self.history_table.setSortingEnabled(True)
        self.history_table.setUpdatesEnabled(True)
    
# ═════════════════════════════════════════════
#  PANEL 3 — ADD STUDENT
# ═════════════════════════════════════════════
class AddStudentPanel(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(16)

        self.page_title = QLabel("Add Student")
        main_w.addWidget(self.page_title)

        # ── Two-column layout: form left, camera right ────
        columns = QHBoxLayout()
        columns.setSpacing(24)

        # Left: form
        form_frame = QFrame()
        form_layout = QFormLayout(form_frame)
        form_layout.setSpacing(12)

        self.name_input     = QLineEdit()
        self.lastname_input = QLineEdit()
        self.section_input  = QLineEdit()
        self.name_input.setMaximumWidth(220)
        self.lastname_input.setMaximumWidth(220)
        self.section_input.setMaximumWidth(220)

        form_layout.addRow("First name ",  self.name_input)
        form_layout.addRow("Last name ",   self.lastname_input)
        form_layout.addRow("Section",       self.section_input)
        columns.addWidget(form_frame)

        # Right: camera preview + capture controls
        cam_frame = QFrame()
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setAlignment(Qt.AlignTop)

        self.cam_label = QLabel("Camera feed")
        self.cam_label.setFixedSize(300, 240)
        self.cam_label.setStyleSheet("background-color: #121212;")
        self.cam_label.setAlignment(Qt.AlignCenter)

        self.capture_progress = QLabel("Samples captured: 0 / 5")
        self.capture_btn      = QPushButton("Start Capture")
        self.redo_btn         = QPushButton("Redo")
        self.redo_btn.setEnabled(False)

        capture_row = QHBoxLayout()
        capture_row.addWidget(self.capture_btn)
        capture_row.addWidget(self.redo_btn)

        cam_layout.addWidget(self.cam_label)
        cam_layout.addWidget(self.capture_progress)
        cam_layout.addLayout(capture_row)
        columns.addWidget(cam_frame)

        main_w.addLayout(columns)

        # ── Status label ─────────────────────────────────
        self.status_lbl = QLabel("")
        main_w.addWidget(self.status_lbl)

        # ── Save / Cancel ────────────────────────────────
        main_w.addStretch()
        btn_row = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn   = QPushButton("Save Student")
        self.save_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.save_btn)
        main_w.addLayout(btn_row)

    def on_enter(self):
        self._reset()

    def _reset(self):
        self.name_input.clear()
        self.lastname_input.clear()
        self.section_input.clear()
        self.capture_progress.setText("Samples captured: 0 / 5")
        self.status_lbl.setText("")
        self.save_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)


# ═════════════════════════════════════════════
#  PANEL 4 — IMPORT FOLDER
# ═════════════════════════════════════════════
class ImportFolderPanel(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(16)

        self.page_title = QLabel("Import Folder")
        main_w.addWidget(self.page_title)

        self.info_lbl = QLabel(
            "Select a folder of photos. Expected filename format: "
            "firstname_lastname_section.jpg")
        self.info_lbl.setWordWrap(True)
        main_w.addWidget(self.info_lbl)

        # ── Folder picker ────────────────────────────────
        picker_row = QHBoxLayout()
        self.path_field = QLineEdit()
        self.path_field.setPlaceholderText("No folder selected...")
        self.path_field.setReadOnly(True)
        self.browse_btn = QPushButton("Browse...")
        picker_row.addWidget(self.path_field)
        picker_row.addWidget(self.browse_btn)
        main_w.addLayout(picker_row)
        
        # ── Preview table ─────────────────────────────────
        self.preview_table = QTableWidget(0, 4)
        self.preview_table.setHorizontalHeaderLabels(
            ["Filename", "Name", "Lastname", "Section"])
        row = self.preview_table.rowCount()
        self.preview_table.insertRow(row)

        self.preview_table.setItem(row, 0, QTableWidgetItem("folder 1"))
        self.preview_table.setItem(row, 1, QTableWidgetItem("Ahmed"))
        self.preview_table.setItem(row, 2, QTableWidgetItem("Sonbol"))
        self.preview_table.setItem(row, 3, QTableWidgetItem("211"))

        row+=1
        self.preview_table.insertRow(row)

        self.preview_table.setItem(row, 0, QTableWidgetItem("folder 2"))
        self.preview_table.setItem(row, 1, QTableWidgetItem("Ahmed"))
        self.preview_table.setItem(row, 2, QTableWidgetItem("Sonbol"))
        self.preview_table.setItem(row, 3, QTableWidgetItem("211"))

        self.preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        main_w.addWidget(self.preview_table)

        # ── Progress ─────────────────────────────────────
        self.progress_label = QLabel("Ready")
        self.progress_bar   = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_w.addWidget(self.progress_label)
        main_w.addWidget(self.progress_bar)

        # ── Buttons ───────────────────────────────────────
        btn_row = QHBoxLayout()
        self.import_btn = QPushButton("Import All")
        self.import_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.import_btn)
        main_w.addLayout(btn_row)

        self.browse_btn.clicked.connect(self._browse)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select photo folder")
        if folder:
            self.path_field.setText(folder)
            self.import_btn.setEnabled(True)

    def on_enter(self):
        pass


# ═════════════════════════════════════════════
#  PANEL 5 — SESSION SETUP
# ═════════════════════════════════════════════
class SessionPanel(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(16)

       

        # ── Splitter: config left | student list right ────
        splitter = QSplitter(Qt.Horizontal)

        # Left: session config form
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)

        session_group = QGroupBox("Session Details")
        session_group.setMaximumHeight(180)
        session_group.setFixedWidth(300)
        
        form = QFormLayout(session_group)
        form.setSpacing(10)

        self.date_picker   = QDateEdit(QDate.currentDate())
        self.date_picker.setCalendarPopup(True)
        self.start_time    = QTimeEdit(QTime(8, 0))
        self.end_time      = QTimeEdit(QTime(17, 0))
        self.shift_name    = QLineEdit()
        self.shift_name.setPlaceholderText("e.g. Morning, Group A...")

        form.addRow("Date:",        self.date_picker)
        form.addRow("Start shift:", self.start_time)
        form.addRow("End shift:",   self.end_time)
        form.addRow("Shift name:",  self.shift_name)
        config_layout.addWidget(session_group)

        # CSV import shortcut
        csv_group = QGroupBox("Import Authorized List")
        csv_group.setMaximumHeight(160)
        csv_group.setFixedWidth(300)
        csv_layout = QVBoxLayout(csv_group)
        self.csv_info = QLabel("Or import from CSV: id, name, lastname")
        self.csv_info.setWordWrap(True)
        csv_layout.addWidget(self.csv_info)
        csv_row = QHBoxLayout()
        self.csv_path   = QLineEdit()
        self.csv_path.setReadOnly(True)
        self.csv_path.setPlaceholderText("No file selected...")
        self.csv_browse = QPushButton("Browse CSV")
        csv_row.addWidget(self.csv_path)
        csv_row.addWidget(self.csv_browse)
        csv_layout.addLayout(csv_row)
        self.csv_import_btn = QPushButton("Import from CSV")
        self.csv_import_btn.setEnabled(False)
        csv_layout.addWidget(self.csv_import_btn)
        config_layout.addWidget(csv_group)

        
        self.create_session_btn = QPushButton("Create Session")
        config_layout.addWidget(self.create_session_btn)
        splitter.addWidget(config_widget)

        # Right: student selection
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        sel_header = QHBoxLayout()
        sel_header.addWidget(QLabel("Authorize students"))
        self.auth_count_lbl = QLabel("0 selected")
        sel_header.addStretch()
        sel_header.addWidget(self.auth_count_lbl)
        selection_layout.addLayout(sel_header)

        self.student_search = QLineEdit()
        self.student_search.setPlaceholderText("Search students...")
        selection_layout.addWidget(self.student_search)

        self.student_list = QTableWidget(0, 2)

        self.student_list.setHorizontalHeaderLabels(["Name", "Section"])
        # testing row 
        row = self.student_list.rowCount()
        self.student_list.setItem(row,0,QTableWidgetItem("Ahmed Sonbol"))
        self.student_list.setItem(row,1,QTableWidgetItem("333"))
        self.student_list.setItem(row+1,0,QTableWidgetItem("Yagoub gamar eddin debyaza"))
        self.student_list.setItem(row,1,QTableWidgetItem("323"))
        self.student_list.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.student_list.setColumnWidth(0, 36)
        self.student_list.verticalHeader().setVisible(True)
        self.student_list.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        selection_layout.addWidget(self.student_list)

        select_btns = QHBoxLayout()
        self.select_all_btn   = QPushButton("Select all")
        self.deselect_all_btn = QPushButton("Deselect all")
        select_btns.addWidget(self.select_all_btn)
        select_btns.addWidget(self.deselect_all_btn)
        select_btns.addStretch()
        selection_layout.addLayout(select_btns)
        splitter.addWidget(selection_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_w.addWidget(splitter)

    def on_enter(self):
        pass


# ═════════════════════════════════════════════
#  PANEL 6 — HISTORY
# ═════════════════════════════════════════════
class HistoryPanel(QWidget):
    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self._build()

    def _build(self):
        main_w = QVBoxLayout(self)
        main_w.setContentsMargins(24, 24, 24, 24)
        main_w.setSpacing(12)

        # ── Header ───────────────────────────────────────
        header = QHBoxLayout()
        self.page_title = QLabel("History")
        self.export_btn = QPushButton("Export CSV")
        header.addWidget(self.page_title)
        header.addStretch()
        header.addWidget(self.export_btn)
        main_w.addLayout(header)

        # ── Filter bar ───────────────────────────────────
        filter_bar = QHBoxLayout()

        self.from_date = QDateEdit(QDate.currentDate())
        self.from_date.setFixedWidth(120)
        self.from_date.setCalendarPopup(True)
        self.to_date   = QDateEdit(QDate.currentDate())
        self.to_date.setFixedWidth(120)

        self.to_date.setCalendarPopup(True)

        self.event_filter = QComboBox()
        self.event_filter.addItems(["All events", "Entry", "Exit", "Alerts Only"])

    
        filter_bar.addWidget(QLabel("From:"))
        filter_bar.addWidget(self.from_date)
        filter_bar.addWidget(QLabel("To:"))
        filter_bar.addWidget(self.to_date)
        filter_bar.addWidget(self.event_filter)
        filter_bar.addStretch()
        main_w.addLayout(filter_bar)

        # ── Log table ────────────────────────────────────
        self.log_table = QTableWidget(0, 6)
        self.log_table.setHorizontalHeaderLabels(
            [ "Student","Date", "Time", "Event", "Confidence", "Alert"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.log_table.setSortingEnabled(True)
        main_w.addWidget(self.log_table)

       
        bot_bar = QHBoxLayout()
        self.count_lbl = QLabel("0 records")
        bot_bar.addWidget(self.count_lbl)
        bot_bar.addStretch()
      
        main_w.addLayout(bot_bar)
        #signals :

        self.export_btn.clicked.connect(self.export_csv)
        self.from_date.dateChanged.connect(self.apply_filter)
        self.to_date.dateChanged.connect(self.apply_filter)
        self.event_filter.currentTextChanged.connect(self.apply_filter)
       

    def on_enter(self):
        self.populate_table(self.db.history_logs)
        

    def populate_table(self,history_logs) : 
        self.log_table.setRowCount(0)
        self.log_table.setSortingEnabled(False)
        self.log_table.setUpdatesEnabled(False)

        for record in history_logs :

            row = self.log_table.rowCount()
            self.log_table.insertRow(row)
            for col , item in enumerate(record) :
                self.log_table.setItem(row,col,QTableWidgetItem(str(item)))
                
        self.log_table.setSortingEnabled(True)
        self.log_table.setUpdatesEnabled(True)
        self.count_lbl.setText(f"{self.log_table.rowCount()} records")
    def apply_filter(self):
        f_date = self.from_date.date().toString("yyyy-MM-dd")
        t_date = self.to_date.date().toString("yyyy-MM-dd")
        event = self.event_filter.currentText().lower()

        matches = []

        for record in self.db.history_logs:

            log_date = str(record[1])
            log_event = str(record[3]).lower()
            alert_type = str(record[5]).lower()

            # Date filter
            if not (f_date <= log_date <= t_date):
                continue

            # Event filter
            if event == "alerts only":
                if alert_type == "none":
                    continue

            elif event != "all events":
                if log_event != event:
                    continue

            matches.append(record)

        self.populate_table(matches)
    def export_csv(self) :
        path,_ = QFileDialog.getSaveFileName(self,"Export csv" ,"report.csv",filter="*.csv")
        if not path : 
            return 
        
        with open(path,"w",newline="\n",encoding="utf-8") as file : 
            writer = csv.writer(file)


            #headers : 

            headers = []

            for col in range(self.log_table.columnCount()) : 

                headers.append(self.log_table.horizontalHeaderItem(col).text())
            writer.writerow(headers)
            # data : 
            
            for row in range(self.log_table.rowCount()) : 
                data = []
                for col in range(self.log_table.columnCount()) :
                    item = self.log_table.item(row,col)

                    data.append(item.text())

                writer.writerow(data) 
        QMessageBox.information(self,"Info","File exported successfuly")


# ═════════════════════════════════════════════
#  AUTH DIALOG  (password gate)
# ═════════════════════════════════════════════
class AuthDialog(QDialog):
    """Simple password confirmation before destructive actions."""
    def __init__(self, action_label="proceed", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Authentication required")
        self.setFixedSize(320, 160)

        main_w = QVBoxLayout(self)
        main_w.setSpacing(12)

        self.info_lbl = QLabel(f"Enter admin password to {action_label}:")
        self.info_lbl.setWordWrap(True)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("Password")

        btn_row = QHBoxLayout()
        self.cancel_btn  = QPushButton("Cancel")
        self.confirm_btn = QPushButton("Confirm")
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.confirm_btn)
        self.confirm_btn.setEnabled(False)

        main_w.addWidget(self.info_lbl)
        main_w.addWidget(self.password_input)
        main_w.addLayout(btn_row)

        self.cancel_btn.clicked.connect(self.reject)
        self.confirm_btn.clicked.connect(self.accept)
        self.password_input.textChanged.connect(self.handletextchange)

    def handletextchange(self,currenttext:str) :
            if not currenttext : 
                self.confirm_btn.setEnabled(False)
                return
            self.confirm_btn.setEnabled(True)
    def get_password(self):
        return self.password_input.text().strip()


# ═════════════════════════════════════════════
#  MAIN DATABASE WINDOW
# ═════════════════════════════════════════════
class DatabaseWindow(QMainWindow):
    requested_panel = Signal(int)

    def __init__(self, db=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Database Manager")
        self.resize(1100, 550)
        self._build()
        self._connect_signals()
        # Show dashboard by default
        self._switch_panel(0)
        theme = self.load_stylesheet("C:/Users/chiko/access_sys/AMS/assets/styles/anotherdbstyle.qss")
        self.setStyleSheet(theme)
        

    # ─────────────────────────────────────────
    #  BUILD
    # ─────────────────────────────────────────
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_w_layout = QHBoxLayout(central)
        main_w_layout.setContentsMargins(0, 0, 0, 0)
        main_w_layout.setSpacing(0)

        main_w_layout.addWidget(self._build_sidebar())
        main_w_layout.addWidget(self._build_content())

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(4)

        # App title
        self.app_title = QLabel("Attendance DB")
        self.app_title.setObjectName("app_title")
        layout.addWidget(self.app_title)
        layout.addSpacing(8)

    
        
        layout.addSpacing(8)

        # Nav buttons — order matches QStackedWidget index
        self.nav_buttons = []
        nav_items = [
            ("Dashboard",     "📊"),
            ("Students",      "👥"),
            ("Add Student",   "➕"),
            ("Import Folder", "📁"),
            ("Session Setup", "📅"),
            ("History",       "🕓"),
        ]
        for i, (label, icon) in enumerate(nav_items):
            btn = NavButton(f"  {icon}  {label}")
            btn.clicked.connect(self.on_nav_clicked)
            self.nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Bottom: user info
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(separator)

        self.user_name_lbl = QLabel("Admin")
        self.user_role_lbl = QLabel("Administrator")
        layout.addWidget(self.user_name_lbl)
        layout.addWidget(self.user_role_lbl)

        return sidebar
    def on_nav_clicked(self,checked) : 
        index = self.nav_buttons.index(self.sender())
        self.requested_panel.emit(index)
        
    def _build_content(self):
        self.stack = QStackedWidget()
        self.panels : list[QWidget]= [
            DashboardPanel(self.db),
            StudentsPanel(self.db),
            AddStudentPanel(self.db),
            ImportFolderPanel(self.db),
            SessionPanel(self.db),
            HistoryPanel(self.db),
        ]
        for panel in self.panels:
            self.stack.addWidget(panel)
        return self.stack

    # ─────────────────────────────────────────
    #  NAVIGATION
    # ─────────────────────────────────────────
    def _switch_panel(self, index:int):
        # Update button checked states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        # Switch stack
        self.stack.setCurrentIndex(index)
        # Notify panel it's now visible
        self.panels[index].on_enter()

    # ─────────────────────────────────────────
    #  SIGNALS
    # ─────────────────────────────────────────
    def _connect_signals(self):
      
        # AddStudentPanel — cancel goes back to students
        ap = self.panels[2]
        ap.cancel_btn.clicked.connect(lambda: self._switch_panel(1))

        # Dashboard — view all logs goes to history
        dp = self.panels[0]
        dp.view_all_btn.clicked.connect(lambda: self._switch_panel(5))

        self.requested_panel.connect(self._switch_panel)

  

    def load_stylesheet(self,filename) : 
        try : 
            with open(filename,"r",encoding="utf-8") as f : 
                return f.read()
        except FileNotFoundError :
            QMessageBox.warning(self,"warning","Could not load style !")
            return ""

