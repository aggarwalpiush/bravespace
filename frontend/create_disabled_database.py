import sqlite3

con = sqlite3.connect('disabled.db')
cur = con.cursor()
cur.execute('CREATE TABLE disabled (id INTEGER PRIMARY KEY, img_name TEXT NOT NULL UNIQUE, hate_score NUMERIC)')
cur.execute("INSERT INTO disabled VALUES (1, '68324.png', 0.98)")
cur.execute("INSERT INTO disabled VALUES (2, '90625.png', 0.69)")
cur.execute("INSERT INTO disabled VALUES (3, '98670.png', 0.87)")
con.commit()
con.close()