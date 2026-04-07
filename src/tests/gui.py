import sys
import os
import cv2
import json
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QTableWidget,
                               QTableWidgetItem, QMessageBox, QGroupBox, 
                               QInputDialog, QLineEdit, QHeaderView, QFrame, QDialog)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap

# --- CONFIG & PATHS ---
if getattr(sys, 'frozen', False):
    app_path = os.path.dirname(sys.executable)
else:
    app_path = os.path.dirname(os.path.abspath(__file__))

DB_FILE = os.path.join(app_path, "face_db.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- HIGH-PERFORMANCE MATCHER ---
class VectorMatcher:
    """Uses matrix operations for O(1) matching speed across large datasets."""
    def __init__(self, db_dict):
        self.names = []
        self.matrix = None
        self.threshold = 0.65 # Confidence threshold
        self.update_db(db_dict)

    def update_db(self, db_dict):
        self.names = []
        embeddings = []
        for name, info in db_dict.items():
            for emb in info.get("embeddings", []):
                self.names.append(name)
                embeddings.append(emb)

        if embeddings:
            self.matrix = np.array(embeddings)
            # Pre-normalize for fast Cosine Similarity
            norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            self.matrix = self.matrix / (norms + 1e-10)
        else:
            self.matrix = None

    def find_match(self, input_emb):
        if self.matrix is None: return "UNKNOWN"
        input_emb = input_emb.flatten()
        input_vec = input_emb / (np.linalg.norm(input_emb) + 1e-10)
        
        # Matrix multiplication = Batch Cosine Similarity
        scores = np.dot(self.matrix, input_vec)
        idx = np.argmax(scores)
        
        return self.names[idx] if scores[idx] > self.threshold else "UNKNOWN"

# --- CORE ENGINE: VIDEO & AI ---
class AIServerThread(QThread):
    frame_ready = Signal(QImage)
    status_update = Signal(str)
    embedding_captured = Signal(np.ndarray) # Safe signal for biometric data
    
    def __init__(self, matcher):
        super().__init__()
        self._active = True
        self.matcher = matcher
        self.yolo = None
        self.facenet = None
        self.frame_count = 0 
        self.cached_names = [] # Store names between recognition frames
    @torch.inference_mode()
    def run(self):
        self.status_update.emit("Loading Neural Engines...")
        try:
            # Ensure you have a model. using 'yolov8n.pt' as fallback if yours is missing
            model_name = "yolov8nbest.pt" # Change this back to your custom model if needed
            self.yolo = YOLO(model_name).to(DEVICE)
            self.facenet = InceptionResnetV1(pretrained='vggface2').to(DEVICE).eval()
            if DEVICE.type == 'cuda': self.facenet.half()
        except Exception as e:
            self.status_update.emit(f"Critical Error: {e}")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_update.emit("Error: Camera not found")
            return

        self.status_update.emit(f"System Online [{DEVICE.type.upper()}]")

        while self._active:
            ret, frame = cap.read()
            if not ret: break
            
            self.frame_count += 1
            
            # 1. Detection Phase (YOLOv8)
            # Detect every frame for smoothness
            results = self.yolo(frame, imgsz=320, conf=0.5, verbose=False, classes=[0])[0] # classes=[0] for person
            
            face_crops = []
            boxes = []
            
            # Extract boxes
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                # Simple heuristic: Only treat as face if size is reasonable
                if (x2-x1) > 20 and (y2-y1) > 20: 
                    face_crops.append(frame[y1:y2, x1:x2])
                    boxes.append((x1, y1, x2, y2))

            # 2. Recognition Phase (Run every 3rd frame to save FPS)
            if self.frame_count % 3 == 0:
                self.cached_names = []
                if face_crops:
                    try:
                        tensors = []
                        for f in face_crops:
                            f_rgb = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (160,160))
                            tensors.append((f_rgb.astype(np.float32)/255.0 - 0.5)/0.5)
                        
                        batch = torch.from_numpy(np.array(tensors)).permute(0, 3, 1, 2).to(DEVICE)
                        if DEVICE.type == 'cuda': batch = batch.half()
                        
                        embs = self.facenet(batch).float().cpu().numpy()
                        
                        # Emit the first found face embedding for registration purposes
                        if len(embs) > 0:
                            self.embedding_captured.emit(embs[0])

                        for emb in embs:
                            name = self.matcher.find_match(emb)
                            self.cached_names.append(name)
                    except:
                        self.cached_names = ["ERROR"] * len(boxes)
                else:
                    self.cached_names = []
            
            # 3. Drawing Phase (Use cached names for stability)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Fallback if cache length mismatch (due to fast movement)
                name = self.cached_names[i] if i < len(self.cached_names) else "Scanning..."
                
                color = (46, 204, 113) if name != "UNKNOWN" and name != "Scanning..." else (231, 76, 60)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color[::-1], 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color[::-1], 1)

            # 4. Output Phase
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            q_img = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            self.frame_ready.emit(q_img)

        cap.release()

    def stop(self):
        self._active = False
        self.wait()

# --- RECORDS MANAGER ---
class RecordsWindow(QDialog):
    db_changed = Signal()

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Authorized Personnel Database")
        self.resize(600, 450)
        self.db = db
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Employee ID", "Full Name"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.populate()
        
        layout.addWidget(self.table)

        btn_bar = QHBoxLayout()
        del_btn = QPushButton("🗑 Delete Entry")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self.delete_selected)
        
        close_btn = QPushButton("Done")
        close_btn.clicked.connect(self.accept)
        
        btn_bar.addWidget(del_btn)
        btn_bar.addStretch()
        btn_bar.addWidget(close_btn)
        layout.addLayout(btn_bar)

    def populate(self):
        self.table.setRowCount(0)
        for name, info in self.db.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(info["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(name))

    def delete_selected(self):
        row = self.table.currentRow()
        if row < 0: return
        
        name = self.table.item(row, 1).text()
        uid = self.table.item(row, 0).text()
        
        confirm = QMessageBox.warning(self, "Security Alert", 
                                     f"Remove {name} (ID: {uid}) from Authorized Access?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if confirm == QMessageBox.Yes:
            if name in self.db:
                del self.db[name]
                self.populate()
                self.db_changed.emit()

# --- MAIN DASHBOARD ---
class VisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VisionID AI Security Dashboard")
        self.setMinimumSize(1200, 800)
        
        self.db = self.load_data()
        self.matcher = VectorMatcher(self.db)
        self.latest_embedding = None # Store latest face vector
        self.worker = None # Initialize worker container
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # Left: Main Feed
        self.feed_lbl = QLabel("SYSTEM IDLE")
        self.feed_lbl.setAlignment(Qt.AlignCenter)
        self.feed_lbl.setScaledContents(True)
        self.feed_lbl.setStyleSheet("background: #000; color: #555; font-size: 18px;")
        main_layout.addWidget(self.feed_lbl, 4)

        # Right: Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background: #1e1e2d; border-left: 1px solid #333;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(20, 30, 20, 30)

        # Title
        header = QLabel("COMMAND CENTER")
        header.setStyleSheet("font-weight: bold; color: #727cf5; font-size: 14px; letter-spacing: 1px;")
        side_layout.addWidget(header)

        # Status Bar
        self.status_lbl = QLabel("STATUS: OFFLINE")
        self.status_lbl.setStyleSheet("color: #fa5c7c; font-size: 11px;")
        side_layout.addWidget(self.status_lbl)
        
        side_layout.addSpacing(30)

        # Group: Identification
        id_group = QGroupBox("Identity Search")
        id_vbox = QVBoxLayout(id_group)
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search Employee ID...")
        id_vbox.addWidget(self.search_bar)
        find_btn = QPushButton("Locate Record")
        find_btn.clicked.connect(self.do_search)
        id_vbox.addWidget(find_btn)
        side_layout.addWidget(id_group)

        # Group: System Operations
        op_group = QGroupBox("System Operations")
        op_vbox = QVBoxLayout(op_group)
        
        self.start_btn = QPushButton("⚡ Initialize System")
        self.start_btn.setStyleSheet("background: #727cf5; color: white; padding: 12px;")
        self.start_btn.setCheckable(True)
        self.start_btn.setChecked(False)
        self.start_btn.clicked.connect(self.boot_shut_system)
        
        add_btn = QPushButton("+ Register New Person")
        add_btn.clicked.connect(self.register_subject)
        
        db_btn = QPushButton("📂 Access Database")
        db_btn.clicked.connect(self.open_db_manager)
        
        op_vbox.addWidget(self.start_btn)
        op_vbox.addWidget(add_btn)
        op_vbox.addWidget(db_btn)
        side_layout.addWidget(op_group)

        side_layout.addStretch()
        
        # Footer
        footer = QLabel(f"Build v1.0.5 | Mode: {DEVICE.type.upper()}")
        footer.setStyleSheet("color: #4b4b5b; font-size: 10px;")
        side_layout.addWidget(footer)

        main_layout.addWidget(sidebar)
        self.apply_style()

    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #1a1a1a; }
            QGroupBox { color: #8e8e93; font-weight: bold; border: 1px solid #333; margin-top: 15px; padding-top: 15px; border-radius: 5px; }
            QPushButton { background: #323248; color: #ffffff; border: none; padding: 8px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background: #3d3d5c; }
            QPushButton#danger { background: #fa5c7c; }
            QLineEdit { background: #2d2d3f; border: 1px solid #333; color: white; padding: 6px; border-radius: 4px; }
        """)

    def load_data(self):
        if not os.path.exists(DB_FILE): return {}
        try:
            with open(DB_FILE, 'r') as f:
                raw = json.load(f)
                return {k: {"id": v["id"], "embeddings": [np.array(e) for e in v["embeddings"]]} for k, v in raw.items()}
        except: return {}

    def save_data(self):
        export = {k: {"id": v["id"], "embeddings": [e.tolist() for e in v["embeddings"]]} for k, v in self.db.items()}
        with open(DB_FILE, 'w') as f: json.dump(export, f, indent=4)
        self.matcher.update_db(self.db)

    # --- FIXED: Proper Thread Lifecycle Management ---
    def boot_shut_system(self):
        if self.start_btn.isChecked():
            # STARTING UP
            if self.worker is not None and self.worker.isRunning():
                return

            self.worker = AIServerThread(self.matcher)
            self.worker.frame_ready.connect(lambda img: self.feed_lbl.setPixmap(QPixmap.fromImage(img)))
            self.worker.status_update.connect(self.update_status)
            self.worker.embedding_captured.connect(self.update_embedding_buffer) # Capture embeddings safely
            
            self.worker.start()
            
            self.start_btn.setText("⛔ DEACTIVATE SYSTEM")
            self.start_btn.setStyleSheet("background: #fa5c7c; color: white; padding: 12px;")
        else:
            # SHUTTING DOWN
            if self.worker:
                self.status_lbl.setText("STATUS: STOPPING...")
                self.worker.stop()
                self.worker = None
            
            self.feed_lbl.clear()
            self.feed_lbl.setText("SYSTEM IDLE")
            self.start_btn.setText("⚡ Initialize System")
            self.start_btn.setStyleSheet("background: #727cf5; color: white; padding: 12px;")

    def update_status(self, txt):
        self.status_lbl.setText(f"STATUS: {txt}")
        color = "#0acf97" if "Online" in txt else "#fa5c7c"
        self.status_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def update_embedding_buffer(self, emb):
        self.latest_embedding = emb

    def register_subject(self):
        # Only allow registration if system is running and we have a face
        if not self.worker or not self.worker.isRunning():
             QMessageBox.critical(self, "System Offline", "Please initialize the system first.")
             return

        if self.latest_embedding is None:
            QMessageBox.critical(self, "Biometric Failure", "No face detected. Please stand in front of the camera.")
            return
        
        # Pause video feed logic logically (optional, here we just grab data)
        name, ok1 = QInputDialog.getText(self, "Registration", "Full Legal Name:")
        if not ok1 or not name: return
        
        uid, ok2 = QInputDialog.getText(self, "Registration", "Unique Employee ID:")
        if not ok2 or not uid: return
        
        if name not in self.db: self.db[name] = {"id": uid, "embeddings": []}
        
        # Save the currently buffered embedding
        self.db[name]["embeddings"].append(self.latest_embedding)
        self.save_data()
        
        self.latest_embedding = None # Clear buffer
        QMessageBox.information(self, "Success", f"Subject '{name}' added to Secure Registry.")

    def do_search(self):
        val = self.search_bar.text().strip()
        for name, info in self.db.items():
            if info["id"] == val:
                QMessageBox.information(self, "Match Found", f"Identity: {name}\n")
                return
        QMessageBox.warning(self, "No Match", "ID not found in registry.")

    def open_db_manager(self):
        win = RecordsWindow(self.db, self)
        win.db_changed.connect(self.save_data)
        win.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = VisionApp()
    window.show()
    sys.exit(app.exec())