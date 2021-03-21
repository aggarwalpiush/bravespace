import sqlite3

con = sqlite3.connect('random.db')
cur = con.cursor()
cur.execute('CREATE TABLE random (id INTEGER PRIMARY KEY, img_name TEXT NOT NULL UNIQUE, hate_score NUMERIC)')
cur.execute("INSERT INTO random VALUES (1, '51682_demo.png', 0.8)")
cur.execute("INSERT INTO random VALUES (2, '52091_demo.png', 0.7)")
cur.execute("INSERT INTO random VALUES (3, '61570_demo.png', 0.6)")
con.commit()
con.close()