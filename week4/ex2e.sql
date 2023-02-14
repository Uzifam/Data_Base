CREATE TABLE Specialization(
  name varchar(10),
  PRIMARY KEY (name) 
);

CREATE TABLE Course(
  name varchar(10),
  credits integer,
  PRIMARY KEY (name)
);

CREATE TABLE Students(
  id integer,
  name varchar(10),
  native_lang varchar(17),
  name_spec varchar(10) not NULL,
  name_course varchar(10) not NULL,
  PRIMARY KEY (id),
  FOREIGN KEY (name_spec) references Specialization(name),
  FOREIGN KEY (name_course) references Course(name)
 );
 
 INSERT INTO Specialization VALUES ('ML');
 INSERT INTO Course VALUES ('Math', 123);
  INSERT INTO Course VALUES ('English', 123);
  INSERT INTO Specialization VALUES ('AI');
 INSERT INTO Course VALUES ('AI', '2');
   INSERT INTO Specialization VALUES ('Robotics');
 INSERT INTO Course VALUES ('Robotics', 123);
 
 INSERT INTO Students VALUES (1, 'Pavel', 'Russian', 'ML', 'Math');
 INSERT INTO Students VALUES (2, 'Paefvel', 'Russaian', 'AI', 'AI');
 INSERT INTO Students VALUES (3, 'Paefvael', 'English', 'Robotics', 'Robotics');





Select Course.name
from Students join Course on Students.name_course = Course.name
where Students.native_lang = 'English'
