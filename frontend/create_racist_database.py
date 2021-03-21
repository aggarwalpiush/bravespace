import sqlite3

con = sqlite3.connect('racist.db')
cur = con.cursor()
cur.execute('CREATE TABLE racist (id INTEGER PRIMARY KEY, img_name TEXT NOT NULL UNIQUE, hate_score NUMERIC)')
cur.execute("INSERT INTO racist VALUES (1, '04356.png', 0.9)")
cur.execute("INSERT INTO racist VALUES (2, '51706.png', 0.7)")
cur.execute("INSERT INTO racist VALUES (3, '61570.png', 0.8)")
con.commit()
con.close()