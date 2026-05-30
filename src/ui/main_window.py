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
from databaseSection.dbwindow import DatabaseWindow
from databaseSection.DatabaseManager import DatabaseManager
from util.utilFuncs import load_data
import onnxruntime as ort
from pathlib import Path 



DEVICE = "CUDA" if "CUDAExecutionProvider" in ort.get_available_providers() else "CPU"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "db" / "database.db"
STYLES_DIR = BASE_DIR / "assets" / "styles"
DARK_MODE_PATH = STYLES_DIR / "dark_style.qss"
LIGHT_MODE_PATH = STYLES_DIR / "light_style.qss"

class MainWindow(QMainWindow) :
    start_camera_sig = Signal()
    def __init__(self) : 
        super().__init__()
        self.setWindowTitle("VisionID Dashboard")
        self.setMinimumSize(1024,800)
        self.init_ui()
        self.worker = None
        self.db = DatabaseManager(DB_PATH)
        self.dbwindow = None
        
        
 
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
        
        self.dark_theme = self.load_stylesheet(DARK_MODE_PATH)
        self.light_theme = self.load_stylesheet(LIGHT_MODE_PATH)
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
        self.db_manager_btn = QPushButton("Open database manager")
        self.db_ctrl_lt.addWidget(self.db_manager_btn)
       
       
        self.side_bar_lt.addWidget(self.db_ctrl)



        self.Attendace_ctrl = QGroupBox("Attendance Control")
        self.Attendace_ctrl_lt = QVBoxLayout(self.Attendace_ctrl)
        self.session_stats_btn = QPushButton("Session stats")
        self.Attendace_ctrl_lt.addWidget(self.session_stats_btn)

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
        self.db_manager_btn.clicked.connect(self.view_databse)
       

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
    
   

   
    def view_databse(self) : 
        if self.dbwindow is None : 
            self.dbwindow =DatabaseWindow(self.db)
            self.dbwindow.destroyed.connect(lambda: setattr(self, "dbwindow", None)  )
        if self.dbwindow.isMinimized():
            self.dbwindow.setWindowState(self.dbwindow.windowState() & ~Qt.WindowMinimized)
        self.dbwindow.show()
        self.dbwindow.raise_()
        self.dbwindow.activateWindow()

   