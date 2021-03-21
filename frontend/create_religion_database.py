import sqlite3

con = sqlite3.connect('religion.db')
cur = con.cursor()
cur.execute('CREATE TABLE religion (id INTEGER PRIMARY KEY, img_name TEXT NOT NULL UNIQUE, hate_score NUMERIC)')
cur.execute("INSERT INTO religion VALUES (1, '51682.png', 0.7)")
cur.execute("INSERT INTO religion VALUES (2, '57982.png', 0.6)")
cur.execute("INSERT INTO religion VALUES (3, '69074.png', 0.98)")
con.commit()
con.close()