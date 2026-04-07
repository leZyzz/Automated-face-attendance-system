from PySide6.QtWidgets import ( QVBoxLayout,
                               QHBoxLayout, QPushButton, QTableWidget,
                               QTableWidgetItem, QMessageBox, QLineEdit,  QDialog,QAbstractItemView,QHeaderView
                               )
from PySide6.QtCore import  Signal


class database_window(QDialog) : 
    db_changed = Signal(dict)

    def __init__(self,db) : 
        super().__init__()
        self.db = db
        self.init_ui()

    def init_ui(self) : 
        self.resize(800,600)
        self.setWindowTitle("DataBase Manager")
        self.main_lt = QVBoxLayout(self)
        self.search_zone = QHBoxLayout()
        self.search_field = QLineEdit(self)
        self.search_btn = QPushButton("Search")
        self.search_zone.addWidget(self.search_field)
        self.search_zone.addWidget(self.search_btn)

        self.search_btn.clicked.connect(self.search)
        self.search_field.textChanged.connect(self.enable_disable_search)
        self.search_field.textChanged.connect(self.search)
        self.main_lt.addLayout(self.search_zone)
        self.table = QTableWidget(0,2)
        self.table.setHorizontalHeaderLabels(["ID","FULL NAME"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows) 
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)

        self.populate(self.db)
        self.main_lt.addWidget(self.table)
        

        self.footer_lt = QHBoxLayout()
        self.delete_btn=QPushButton("Delete Selected")
        self.footer_lt.addWidget(self.delete_btn)
        self.footer_lt.addStretch()
        self.main_lt.addLayout(self.footer_lt)

        self.delete_btn.clicked.connect(self.delete_selected)
    def populate(self,db):
        self.table.setRowCount(0)
        for name, info in db.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(info["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(name))

    def delete_selected(self) : 
        selected_items = self.table.selectedItems()
        if not selected_items : 
            QMessageBox.information(self,"Info" , "Please select a person to delete !")
            return 
        selected_rows = set()
        for item in selected_items :
            selected_rows.add(item.row())

        confirm = QMessageBox.question(self,"Warning", "Carefull This operation cant be undo" , QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.No : 
            return
        for row in sorted(selected_rows,reverse=True) : 
            target_id= self.table.item(row,0).text()
            target_name = self.table.item(row,1).text()

            
            if target_name in self.db : 
                if str(self.db[target_name]["id"]) == target_id : 
                    del self.db[target_name]

            self.table.removeRow(row)


        self.save_db()
        QMessageBox.information(self,"Success" , "Selected profiles deleted successfully")                

    def save_db(self) : 
        self.populate(self.db)
        self.db_changed.emit(self.db)

    def enable_disable_search(self):
        text = self.search_field.text().strip()
        
        self.search_btn.setEnabled(bool(text))
        
        if not text:
            self.populate(self.db)
    def search(self):
        target = self.search_field.text().strip().lower() 
        
        if not target:
            self.populate(self.db)
            return

        match_cases = {}
        for name, info in self.db.items():
            if target in name.lower() or target in str(info['id']).lower():
                match_cases[name] = info  
        if match_cases : 
            self.populate(match_cases)
    
        else:
            self.populate(match_cases)
