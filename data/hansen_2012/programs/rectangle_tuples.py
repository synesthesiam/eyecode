def area(xy_1, xy_2):
    width = xy_2[0] - xy_1[0]
    height = xy_2[1] - xy_1[1]
    return width * height

r1_xy_1 = (0, 0)
r1_xy_2 = (10, 10)
r1_area = area(r1_xy_1, r1_xy_2)
print r1_area

r2_xy_1 = (5, 5)
r2_xy_2 = (10, 10)
r2_area = area(r2_xy_1, r2_xy_2)
print r2_area
