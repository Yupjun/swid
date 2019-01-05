#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##처음에 해야 하는 부분
import sqlite3
from datetime import datetime
import os


class DB_Controller():
    ## initialize
    def __init__(self):
        self.db_list = [f for f in os.listdir("./") if f.endswith('.db')]
        conn = sqlite3.connect("Inspection.db")
        cur = conn.cursor()
        cur.execute('''CREATE TABLE INSPECTION_LOG
             (Date text PRIMARY KEY,Device INTEGER, IDX INTEGER, SIGMOID REAL)''')
        
    def empty(self):
        return self.db_list != []
    
#     #자정이 지났는지 확인한다
#     def check_midnight(self):
#         now = datetime.now()
#         seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
#         return seconds_since_midnight <1000
        
        
# #     #DB 리스트에 이미 오늘자의 DB를 생성했는지 확인한다.
    
#     def check_db_list(self):
#         temps = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         return temps in self.db_list
        

#     def create_db(self):
#         conn = sqlite3.connect(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    
    def commit(self,device,index,sigmoid):
        conn = sqlite3.connect("Inspection.db")
        cur = conn.cursor()
        time = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         conn = sqlite3.connect(TIME, isolation_level=None)
#         cur = conn.cursor()

        tup= str((str(time),str(device),str(index),str(sigmoid)))
        execute = """INSERT INTO INSPECTION_LOG  VALUES """+tup
        cur.execute(execute)
        conn.commit()
        conn.close()
        
        
temp = DB_Controller()
temp.commit(1,2,3)


# In[ ]:





# In[ ]:


conn = sqlite3.connect()


# In[ ]:





# In[2]:


a = ('01-01','01-02','01-03')


# In[4]:


print(str(a))


# In[ ]:


|

