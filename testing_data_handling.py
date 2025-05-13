import sqlite3
import datetime

# Define the database file name
DATABASE_FILE = 'cyclist_data.db'

def connect_db():
    """Connects to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    return conn, cursor

def create_table():
    """Creates the cyclist_counts table if it doesn't exist."""
    conn, cursor = connect_db()
    cursor.execute("\n"
                   "        CREATE TABLE IF NOT EXISTS cyclist_counts (\n"
                   "            timestamp TEXT PRIMARY KEY,\n"
                   "            count INTEGER NOT NULL\n"
                   "        )\n"
                   "    ")
    conn.commit()
    conn.close()

def insert_dummy_count(timestamp_str, count):
    """Inserts a specific timestamp and cyclist count into the database."""
    conn, cursor = connect_db()
    try:
        cursor.execute("INSERT INTO cyclist_counts (timestamp, count) VALUES (?, ?)", (timestamp_str, count))
        conn.commit()
        print(f"Dummy data: {count} cyclists recorded at {timestamp_str}")
    except sqlite3.IntegrityError:
        print(f"Dummy data: Count already recorded for timestamp: {timestamp_str}")
    finally:
        conn.close()

def get_all_counts():
    """Retrieves all recorded counts from the database."""
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM cyclist_counts")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_counts_by_date(date_str):
    """Retrieves counts for a specific date (YYYY-MM-DD)."""
    conn, cursor = connect_db()
    cursor.execute("SELECT * FROM cyclist_counts WHERE strftime('%Y-%m-%d', timestamp) = ?", (date_str,))
    rows = cursor.fetchall()
    conn.close()
    return rows



if __name__ == '__main__':
    create_table() # Ensure the table exists

    dummy_data = [
        ('2025-05-12 08:00:00', 2),
        ('2025-05-12 08:05:30', 1),
        ('2025-05-12 08:12:15', 3),
        ('2025-05-12 12:30:45', 0),
        ('2025-05-12 12:35:20', 1),
        ('2025-05-12 17:10:05', 5),
        ('2025-05-12 17:18:50', 2),
        ('2025-05-12 17:25:30', 4),
    ]

    for timestamp, count in dummy_data:
        insert_dummy_count(timestamp, count)  # Corrected line

    print("\nDummy data insertion complete.")



# if __name__ == '__main__':
#     create_table() # Create the table if it's the first run
#
#     # Example of inserting a count (you would call this from your cyclist detection logic)
#     # Let's say your cyclist counting variable is 'num_cyclists'
#     num_cyclists = 5
#     insert_count(num_cyclists)
#     time.sleep(10) # Simulate another detection after some time
#     num_cyclists = 2
#     insert_count(num_cyclists)
#
#     # Example of retrieving all counts
#     all_data = get_all_counts()
#     print("\nAll recorded data:")
#     for row in all_data:
#         print(row)
#
#     # Example of retrieving counts for today's date
#     today = datetime.date.today().strftime('%Y-%m-%d')
#     today_data = get_counts_by_date(today)
#     print(f"\nCounts for {today}:")
#     for row in today_data:
#         print(row)