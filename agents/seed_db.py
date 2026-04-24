import sqlite3

conn = sqlite3.connect("movies.db")
conn.executescript("""
CREATE TABLE IF NOT EXISTS directors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT,
    birth_year INTEGER
);
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    genre TEXT,
    director_id INTEGER REFERENCES directors(id),
    rating REAL,
    runtime INTEGER,
    box_office INTEGER
);
CREATE TABLE IF NOT EXISTS actors (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    nationality TEXT,
    birth_year INTEGER
);
CREATE TABLE IF NOT EXISTS cast_members (
    movie_id INTEGER REFERENCES movies(id),
    actor_id INTEGER REFERENCES actors(id),
    character_name TEXT,
    PRIMARY KEY (movie_id, actor_id)
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER REFERENCES movies(id),
    reviewer_name TEXT,
    rating INTEGER,
    comment TEXT,
    review_date TEXT
);
""")

conn.executemany("INSERT OR IGNORE INTO directors VALUES (?,?,?,?)", [
    (1, "Christopher Nolan",  "British",    1970),
    (2, "James Cameron",      "Canadian",   1954),
    (3, "Quentin Tarantino",  "American",   1963),
    (4, "Steven Spielberg",   "American",   1946),
    (5, "Denis Villeneuve",   "Canadian",   1967),
])

conn.executemany("INSERT OR IGNORE INTO movies VALUES (?,?,?,?,?,?,?,?)", [
    (1,  "Inception",          2010, "Sci-Fi",    1, 8.8, 148, 836),
    (2,  "Interstellar",       2014, "Sci-Fi",    1, 8.6, 169, 701),
    (3,  "The Dark Knight",    2008, "Action",    1, 9.0, 152, 1005),
    (4,  "Dunkirk",            2017, "War",       1, 7.9, 106, 527),
    (5,  "Avatar",             2009, "Sci-Fi",    2, 7.9, 162, 2923),
    (6,  "Titanic",            1997, "Romance",   2, 7.9, 195, 2257),
    (7,  "Aliens",             1986, "Sci-Fi",    2, 8.4, 137, 183),
    (8,  "Pulp Fiction",       1994, "Crime",     3, 8.9, 154, 214),
    (9,  "Django Unchained",   2012, "Western",   3, 8.4, 165, 425),
    (10, "Inglourious Basterds",2009,"War",       3, 8.3, 153, 321),
    (11, "Schindler's List",   1993, "Drama",     4, 9.0, 195, 322),
    (12, "Jurassic Park",      1993, "Sci-Fi",    4, 8.2, 127, 1046),
    (13, "Blade Runner 2049",  2017, "Sci-Fi",    5, 8.0, 164, 260),
    (14, "Dune",               2021, "Sci-Fi",    5, 8.0, 155, 401),
    (15, "Arrival",            2016, "Sci-Fi",    5, 7.9, 116, 203),
])

conn.executemany("INSERT OR IGNORE INTO actors VALUES (?,?,?,?)", [
    (1,  "Leonardo DiCaprio",  "American",   1974),
    (2,  "Tom Hardy",          "British",    1977),
    (3,  "Cillian Murphy",     "Irish",      1976),
    (4,  "Sam Worthington",    "Australian", 1976),
    (5,  "Kate Winslet",       "British",    1975),
    (6,  "Sigourney Weaver",   "American",   1949),
    (7,  "John Travolta",      "American",   1954),
    (8,  "Samuel L. Jackson",  "American",   1948),
    (9,  "Jamie Foxx",         "American",   1967),
    (10, "Liam Neeson",        "Irish",      1952),
    (11, "Jeff Goldblum",      "American",   1952),
    (12, "Ryan Gosling",       "Canadian",   1980),
    (13, "Harrison Ford",      "American",   1942),
    (14, "Timothée Chalamet",  "American",   1995),
    (15, "Matthew McConaughey","American",   1969),
    (16, "Anne Hathaway",      "American",   1982),
    (17, "Tom Hanks",          "American",   1956),
    (18, "Brad Pitt",          "American",   1963),
])

conn.executemany("INSERT OR IGNORE INTO cast_members VALUES (?,?,?)", [
    # Inception
    (1, 1,  "Dom Cobb"),
    (1, 2,  "Eames"),
    (1, 3,  "Robert Fischer"),
    # Interstellar
    (2, 15, "Cooper"),
    (2, 16, "Brand"),
    (2, 3,  "Mann"),
    # The Dark Knight
    (3, 3,  "Scarecrow"),
    (3, 2,  "Bane"),
    # Dunkirk
    (4, 3,  "Farrier"),
    # Avatar
    (5, 4,  "Jake Sully"),
    (5, 6,  "Dr. Grace Augustine"),
    # Titanic
    (6, 1,  "Jack Dawson"),
    (6, 5,  "Rose DeWitt Bukater"),
    # Aliens
    (7, 6,  "Ellen Ripley"),
    # Pulp Fiction
    (8, 7,  "Vincent Vega"),
    (8, 8,  "Jules Winnfield"),
    # Django Unchained
    (9, 9,  "Django"),
    (9, 8,  "Stephen"),
    (9, 1,  "Calvin Candie"),
    # Inglourious Basterds
    (10, 18, "Lt. Aldo Raine"),
    # Schindler's List
    (11, 10, "Oskar Schindler"),
    # Jurassic Park
    (12, 11, "Dr. Ian Malcolm"),
    (12, 13, "Dr. Alan Grant"),
    # Blade Runner 2049
    (13, 12, "K"),
    (13, 13, "Rick Deckard"),
    # Dune
    (14, 14, "Paul Atreides"),
    (14, 17, "Duke Leto Atreides"),
    # Arrival
    (15, 16, "Louise Banks"),
])

conn.executemany("INSERT OR IGNORE INTO reviews VALUES (?,?,?,?,?,?)", [
    (1, 1,  "Alex",   10, "A masterpiece of layered storytelling.", "2024-01-10"),
    (2, 1,  "Maria",  9,  "Visually stunning, emotionally complex.", "2024-01-15"),
    (3, 3,  "John",   10, "The Joker scenes are unforgettable.",     "2024-02-01"),
    (4, 8,  "Sara",   9,  "Changed cinema forever.",                 "2024-02-10"),
    (5, 11, "Tom",    10, "Devastating and important.",               "2024-03-01"),
    (6, 14, "Emma",   9,  "Epic world-building. Cannot wait for part 2.", "2024-03-15"),
    (7, 2,  "Chris",  9,  "The docking scene broke my brain.",        "2024-04-01"),
    (8, 13, "Nina",   8,  "Gorgeous but slow. Worth every minute.",   "2024-04-10"),
])

conn.commit()
conn.close()
print("movies.db seeded.")
