class Person:
    def __init__(self, startX, startY, startVx, startVy):
        self.x = startX
        self.y = startY
        self.Vx = startVx
        self.Vy = startVy
    float Fx = 0;
    float Fy = 0;
    def getX(self):
        return self.x

    def getY(self):
        return self.y
    
    def getVx(self):
        return self.Vx
    
    def getVy(self):
        return self.Vy
    
    def ApplyForce (self,Fx, Fy)
    
    
    def Update(self,time):  
        self.x = self.x + self.Vx*time + 0.5*self.Fx*time^2
        self.y = self.y + self.Vy*time + 0.5*self.Fy*time^2
        
    

# Create an instance of the Person class
person1 = Person(1, 1,1,1)

# Access getX method on the instance of Person
x_value = person1.getX()
print("X value:", x_value)  # Output: X value: 1

# Move the person
person1.Update(1)

# Retrieve updated position
new_x_value = person1.getX()
new_y_value = person1.getY()
print("New X value:", new_x_value)  # Output: New X value: 3
print("New Y value:", new_y_value)  # Output: New Y value: 4
