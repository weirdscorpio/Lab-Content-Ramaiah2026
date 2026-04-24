import sqlite3

conn = sqlite3.connect("bookstore.db")
conn.executescript("""
CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT,
    birth_year INTEGER
);
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author_id INTEGER REFERENCES authors(id),
    genre TEXT,
    price REAL,
    stock INTEGER,
    published_year INTEGER
);
CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    book_id INTEGER REFERENCES books(id),
    customer_name TEXT,
    quantity INTEGER,
    sale_date TEXT
);
""")

conn.executemany("INSERT OR IGNORE INTO authors VALUES (?,?,?,?)", [
    (1, "George Orwell",               "UK",       1903),
    (2, "J.K. Rowling",                "UK",       1965),
    (3, "Haruki Murakami",             "Japan",    1949),
    (4, "Toni Morrison",               "USA",      1931),
    (5, "Gabriel García Márquez",      "Colombia", 1927),
])

conn.executemany("INSERT OR IGNORE INTO books VALUES (?,?,?,?,?,?,?)", [
    (1,  "1984",                                    1, "Dystopian",         12.99,  45, 1949),
    (2,  "Animal Farm",                             1, "Political Satire",   9.99,  30, 1945),
    (3,  "Harry Potter and the Philosopher's Stone",2, "Fantasy",           14.99, 120, 1997),
    (4,  "Harry Potter and the Chamber of Secrets", 2, "Fantasy",           14.99,  95, 1998),
    (5,  "Norwegian Wood",                          3, "Literary Fiction",  13.99,  28, 1987),
    (6,  "Kafka on the Shore",                      3, "Magical Realism",   15.99,  22, 2002),
    (7,  "Beloved",                                 4, "Historical Fiction",13.49,  18, 1987),
    (8,  "One Hundred Years of Solitude",           5, "Magical Realism",   16.99,  35, 1967),
])

conn.executemany("INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?)", [
    (1,  1, "Alice Chen",    2, "2024-01-15"),
    (2,  3, "Bob Smith",     1, "2024-01-16"),
    (3,  8, "Clara Davis",   3, "2024-01-20"),
    (4,  1, "David Lee",     1, "2024-02-01"),
    (5,  5, "Emma Wilson",   2, "2024-02-10"),
    (6,  3, "Frank Brown",   1, "2024-02-14"),
    (7,  7, "Grace Kim",     1, "2024-03-01"),
    (8,  2, "Henry Zhang",   4, "2024-03-05"),
    (9,  6, "Iris Patel",    1, "2024-03-12"),
    (10, 4, "Jack Turner",   2, "2024-03-20"),
    (11, 3, "Karen White",   1, "2024-04-01"),
    (12, 8, "Liam Scott",    1, "2024-04-05"),
])

conn.commit()
conn.close()
print("bookstore.db seeded.")
