import sqlite3
from pathlib import Path
from PySide6.QtCore import QObject , Signal , Slot




class DatabaseManager(QObject) : 
    students_changed = Signal()
    history_changed  = Signal()
    def __init__(self,db_path:str) : 
        super().__init__()
        self.db = sqlite3.connect(db_path)
        self.students =self.db.execute("select * from students").fetchall()
        self.students_changed.connect(self.refetch_all_students)
        self.history_logs = self.db.execute("""select student_id,log_day,log_time,event_type,confidence,alert_type
                                                from access_logs""").fetchall()
      
        self.history_changed.connect(self.refetch_history_logs)
        self.DB_PASS = "hardpassword"
   
    def refetch_all_students(self) ->list[tuple[str]] : 
        self.students=self.db.execute("select * from students").fetchall()

    def get_embeddingCount(self,student_id:str) -> int:
        return self.db.execute("select count(*) from embeddings where student_id= ?",(student_id,)).fetchone()[0]

    def get_student_history(self, student_id:str) ->  list[tuple[str]] :
            return self.db.execute("""
                SELECT
                    exitt.log_day,
                    exitt.log_time AS exit_time,
                    entry.log_time AS return_time
                FROM access_logs exitt
                LEFT JOIN access_logs entry
                    ON entry.student_id = exitt.student_id
                    AND entry.log_day = exitt.log_day
                    AND entry.event_type = 'entry'
                WHERE exitt.student_id = ?
                    AND exitt.event_type = 'exit'
                ORDER BY exitt.log_day DESC;
            """, (student_id,)).fetchall()
    def delete_student(self,students:list[tuple[str]]) : 
         pass
    def refetch_history_logs(self) : 
         self.history_logs = self.db.execute("""select student_id,log_day,log_time,event_type,confidence,alert_type
                                                from access_logs""").fetchall()
         