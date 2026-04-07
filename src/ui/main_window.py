import os 
import json
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel,
                            QMessageBox, QGroupBox, 
                               QInputDialog,  QFrame
                               ,QFileDialog)
from PySide6.QtCore import QThread, Signal, Qt , Slot 
from PySide6.QtGui import QPixmap
from workers.AiWorker import AiWorker
from databaseSection.database_window import database_window
from torch import device ,cuda
from util.utilFuncs import load_data
DEVICE = device("cuda" if cuda.is_available() else "cpu")
class MainWindow(QMainWindow) :
    start_camera_sig = Signal()
    def __init__(self) : 
        super().__init__()
        self.setWindowTitle("VisionID Dashboard")
        self.setMinimumSize(1024,800)
        self.init_ui()
        self.worker = None
        self.db = load_data("/home/zyzz/py/myenv/data/db_file1.json")
        
        
 
    def boot_shut_sys(self):
        if self.init_sys_btn.isChecked() :  
                self.status_lb.setStyleSheet("")
                self.status_lb.setText("initializing system....")
                self.init_sys_btn.setEnabled(False)
                QApplication.processEvents()

                self.worker = AiWorker(self.db)
                self.Thread = QThread()
                self.worker.moveToThread(self.Thread)

                self.worker.status_update.connect(self.update_status)
                self.worker.frame_ready.connect(self.update_frame)
                self.start_camera_sig.connect(self.worker.start_camera_loop)
                self.Thread.start()
                self.start_camera_sig.emit()

                self.init_sys_btn.setEnabled(True)
                self.status_lb.setText("Status : System active")
                self.status_lb.setStyleSheet("color : lightgreen ")
                self.init_sys_btn.setText("Stop camera")
        else : 
                self.status_lb.setStyleSheet("color : red ")
                self.status_lb.setText("Status : stopping...")
                self.init_sys_btn.setEnabled(False)
                QApplication.processEvents()

                if self.worker : 
                    self.worker.stop()
                    self.Thread.quit()
                    self.Thread.wait()
                    self.worker = None
                    self.Thread = None
            
                self.status_lb.setText("Status : ....")
                self.status_lb.setStyleSheet("")
                self.init_sys_btn.setText("start camera")
                self.feed_lb.clear()
                self.feed_lb.setText("System Idle")
                self.feed_lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.feed_lb.setStyleSheet("""    QLabel#feed_label {
                                            background-color: #000000;
                                            border: 1px solid #333333;
                                            border-radius: 4px; }""")
                self.init_sys_btn.setEnabled(True)
                self.init_sys_btn.setText("Start camera")
     
    def init_ui(self) : 
        self.setWindowTitle("VisionID Dashboard")
        self.setMinimumSize(1024,800)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        self.dark_theme = self.load_stylesheet("/home/zyzz/py/myenv/assets/styles/dark_style.qss")
        self.light_theme = self.load_stylesheet("/home/zyzz/py/myenv/assets/styles/light_style.qss")
        self.current_theme = "dark_theme"
        
        # ---LIVE FEED SIDE---
         
        self.feed_lb = QLabel()
        self.feed_lb.setObjectName("feed_label")
        self.feed_lb.setText("System Idle")
        self.feed_lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed_lb.setScaledContents(True)
        self.main_layout.addWidget(self.feed_lb,stretch=4)
        
        # ---Control panel---
        self.side_bar_frame = QFrame()
        self.side_bar_frame.setFixedWidth(300)
        self.side_bar_lt = QVBoxLayout(self.side_bar_frame)
        self.side_bar_lt.setSpacing(30)
        self.side_bar_lt.setContentsMargins(20, 30, 20, 30)
        
        self.CmdCenter = QLabel("COMMAND CENTER")
        self.CmdCenter.setObjectName("cmdcenter")
        self.CmdCenter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.side_bar_lt.addWidget(self.CmdCenter)
        
        
        self.status_lb = QLabel("Status : ...")
        self.side_bar_lt.addWidget(self.status_lb)
        
        self.side_bar_lt.addSpacing(30)
        
        # ---System Controls--- 
        
        self.sys_ctrl = QGroupBox("System Controls")
        self.sys_ctrl_lt = QVBoxLayout(self.sys_ctrl)
        self.init_sys_btn = QPushButton("Start Camera")
        self.init_sys_btn.setCheckable(True)
        self.sys_ctrl_lt.addWidget(self.init_sys_btn)
        self.side_bar_lt.addWidget(self.sys_ctrl)
        self.side_bar_lt.addSpacing(20)
        #---Database Management---
    
        self.db_ctrl= QGroupBox("Database Management")
        self.db_ctrl_lt = QVBoxLayout(self.db_ctrl)
        self.db_view_btn = QPushButton("View database")
        self.db_ctrl_lt.addWidget(self.db_view_btn)
        self.db_add_btn = QPushButton("Add Person")
        self.db_add_btn.clicked.connect(self.add_person)
        self.db_ctrl_lt.addWidget(self.db_add_btn)
        self.db_import_btn = QPushButton("Import folder")
        self.db_ctrl_lt.addWidget(self.db_import_btn)
  
        self.side_bar_lt.addWidget(self.db_ctrl)



        self.Attendace_ctrl = QGroupBox("Attendance Control")
        self.Attendace_ctrl_lt = QVBoxLayout(self.Attendace_ctrl)
        self.view_list_btn = QPushButton("View Authorized Persons")
        self.Attendace_ctrl_lt.addWidget(self.view_list_btn)

        self.upload_list_btn = QPushButton("Upload List")
        self.Attendace_ctrl_lt.addWidget(self.upload_list_btn)


        self.side_bar_lt.addWidget(self.Attendace_ctrl)
        self.side_bar_lt.addStretch()
        
        self.footer = QHBoxLayout()
        self.footer.setSpacing(10)
        self.build_label = QLabel(f"build v1.0.0 | Mode {DEVICE}")
        self.build_label.setObjectName("buildlabel")
        self.footer.addWidget(self.build_label)
        self.toggle_theme_btn = QPushButton("Light Mode")
        self.footer.addWidget(self.toggle_theme_btn)
        self.toggle_theme_btn.clicked.connect(self.toggle_theme)
        self.side_bar_lt.addLayout(self.footer)
        self.main_layout.addWidget(self.side_bar_frame)
        self.setStyleSheet(self.dark_theme)      
        
        
        
        self.init_sys_btn.clicked.connect(self.boot_shut_sys)
        self.db_view_btn.clicked.connect(self.view_databse)
        self.db_import_btn.clicked.connect(self.import_folder)
    def load_stylesheet(self,filename) : 
        try : 
            with open(filename,"r") as f : 
                return f.read()
        except FileNotFoundError :
            QMessageBox.information(self,"warning","Could not load style !")
            return ""
    def toggle_theme (self) :
            if self.current_theme == "light_theme" :
                self.current_theme ="dark_theme"
                self.toggle_theme_btn.setText("Light Mode")
                self.setStyleSheet(self.dark_theme)   
            else : 
                self.current_theme ="light_theme"
                self.toggle_theme_btn.setText("Dark Mode")
                self.setStyleSheet(self.light_theme)                
   
    def update_frame(self,frame) : 
        self.feed_lb.setPixmap(QPixmap.fromImage(frame))
    def update_status(self,status) : 
        self.status_lb.setText(status)
    
    def closeEvent(self, event):
        serialized_db = {}
        if self.worker is not None : 
            for name,info in self.worker.db.items() : 
                serialized_db[name]={
                    "id" : info["id"],
                    "embeddings" : [emb.tolist() for emb in info["embeddings"]]
                }
            try : 
                with open("face_db1.json" , "w") as f : 
                    json.dump(serialized_db,f,indent=4)
            except : 
                pass
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.stop()
            if hasattr(self, 'Thread') and self.Thread is not None:
                self.Thread.quit()
                self.Thread.wait()
        event.accept()
    
    def add_person(self) : 
        name , ok_pressed = QInputDialog.getText(self,"New regisrtation","Enter name : ")
        if ok_pressed : 
            clean_name = name.strip()
            if clean_name!="" : 
                self.worker.start_registration(clean_name)
            else : 
                QMessageBox.warning(self,"Invalid input" , "Name cannot be empty")

    @Slot(dict)
    def update_db(self,db) : 
        self.db=db 
    def view_databse(self) : 
        db_windown = database_window(self.db)
        old_state = self.init_sys_btn.isChecked() 
        if  old_state :
            self.init_sys_btn.setChecked(False)
            self.boot_shut_sys()
        db_windown.db_changed.connect(self.update_db)
        db_windown.exec()
        if old_state :
            self.init_sys_btn.setChecked(True)
            self.boot_shut_sys()
    def import_folder(self): 
        response = QFileDialog.getExistingDirectoryUrl(
            parent=self,
            caption='Select Folder',
            dir=os.getcwd(),
        )
        QMessageBox.information(self,"response",response.toString()) 
        response = QFileDialog.getExistingDirectoryUrl(
            parent=self,
            caption='Select Folder',
            dir=os.getcwd(),
        )
        QMessageBox.information(self,"response",response.toString())