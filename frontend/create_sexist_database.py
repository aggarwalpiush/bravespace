import sqlite3

con = sqlite3.connect('sexist.db')
cur = con.cursor()
cur.execute('CREATE TABLE sexist (id INTEGER PRIMARY KEY, img_name TEXT NOT NULL UNIQUE, hateful_score NUMERIC)')
cur.execute("INSERT INTO sexist VALUES (1, '49867_demo.png', 0.9)")
cur.execute("INSERT INTO sexist VALUES (2, '52091_demo.png', 0.87)")
cur.execute("INSERT INTO sexist VALUES (3, '59316_demo.png', 0.66)")
con.commit()
con.close()