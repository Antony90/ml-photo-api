import mysql.connector
from mysql.connector import errorcode

DB_NAME = 'faces'

class Database:
  def __init__(self):
    try:
      self.conn = mysql.connector.connect(
        user = 'admin',
        password = 'password',
        host = '127.0.0.1',
      )
    except mysql.connector.Error as e:
      if e.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database not found, creating")
      else:
        print(e)
        
  def close(self):
    self.conn.close()
    
  @staticmethod
  def create(self, cursor):
    try:
      cursor.execute(
        "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    except mysql.connector.Error as err:
      print("Failed creating database: {}".format(err))
      exit(1)

  
  @staticmethod
  def create_tables(self, cursor):
    TABLES = {}
    TABLES['face_encodings'] = (
      "CREATE TABLE `face_encodings` ("
      "`user_id` NOT NULL CHAR(20),"
      "`person_id` NOT NULL int(10)"
      "`img_id` NOT NULL CHAR(20)"
      "`encoding` NOT NULL VARBINARY(20),"
      "PRIMARY KEY (`user_id`, `img_id`),"
      "CONSTRAINT `enc_person_id_fk` FOREIGN KEY (`person_id`)"
      "REFERENCES `people`(`person_id`),"
      "CONSTRAINT `enc_user_id_fk` FOREIGN KEY (`user_id`)"
      "REFERENCES `users`(`user_id`) ON DELETE CASCADE"
    )
    TABLES['users'] = (
      "CREATE TABLE `users` ("
      "`user_id` NOT NULL CHAR(20),"
      "PRIMARY KEY (`user_id`),"
      "ON DELETE CASCADE"
    )
    TABLES['people'] = (
      "CREATE TABLE `people` ("
      "`person_id` int(10) NOT NULL AUTO_INCREMENT"
      "`user_id` NOT NULL CHAR(20),"
      "PRIMARY KEY (`user_id`, `person_id`),"
      "CONSTRAINT `people_user_id_fk` FOREIGN KEY (`user_id`)"
      "REFERENCES `users`(`user_id`) ON DELETE CASCADE"
    )
    
    for table_name, statement in TABLES.keys():
      try:
        print("Creating table {}: ".format(table_name), end='')
        cursor.execute(statement)
      except mysql.connector.Error as err:
        if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
          print("already exists.")
        else:
          print(err.msg)
      else:
        print("OK")
        
        
if __name__ == '__main__':
  db = Database()