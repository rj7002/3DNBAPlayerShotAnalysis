# import pandas as pd
# import numpy as np

# class CourtCoordinates:
#     '''
#     Stores court dimensions and calculates the (x,y,z) coordinates of the outside perimeter, 
#     three-point line, backboard, hoop, and free throw line.
#     '''
#     def __init__(self):
#         # New court dimensions to fit the expanded coordinate ranges
#         self.court_length = 700  # Expanded length to fit y range
#         self.court_width = 500   # Expanded width to fit x range
#         self.hoop_loc_x = 0      # Center of the court for expanded x range
#         self.hoop_loc_y = self.court_length / 2  # Center of the court for expanded y range
#         self.hoop_loc_z = 10    *10 # Height of the hoop
#         self.hoop_radius = .75*10
#         self.three_arc_distance = 23.75*10
#         self.three_straight_distance = 22*10
#         self.three_straight_length = 8.89*10
#         self.backboard_width = 6*10
#         self.backboard_height = 4*10
#         self.backboard_baseline_offset = 3*10
#         self.backboard_floor_offset = 9*10
#         self.free_throw_line_distance = 15*10

#     @staticmethod
#     def calculate_quadratic_values(a, b, c):
#         x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
#         x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
#         return x1, x2

#     def __get_court_perimeter_coordinates(self):
#         width = self.court_width
#         length = self.court_length
#         court_perimeter_bounds = [
#             [-width / 2, -length / 2, 0],
#             [width / 2, -length / 2, 0],
#             [width / 2, length / 2, 0],
#             [-width / 2, length / 2, 0],
#             [-width / 2, -length / 2, 0]
#         ]
#         court_df = pd.DataFrame(court_perimeter_bounds, columns=['x', 'y', 'z'])
#         court_df['line_group'] = 'outside_perimeter'
#         court_df['color'] = 'court'
#         return court_df

#     def __get_half_court_coordinates(self):
#         width = self.court_width
#         half_length = self.court_length / 2
#         circle_radius = 6 * 10
#         circle_radius2 = 2 * 10
#         circle_center = [0, half_length-600, 0]
#         circle_points = []
#         circle_points2 = []
#         num_points = 400
#         for i in range(num_points):
#             angle = 2 * np.pi * i / num_points
#             x = circle_center[0] + circle_radius * np.cos(angle)
#             y = circle_center[1] + circle_radius * np.sin(angle)
#             circle_points.append([x, y, circle_center[2]])
#         for i in range(num_points):
#             angle = 2 * np.pi * i / num_points
#             x = circle_center[0] + circle_radius2 * np.cos(angle)
#             y = circle_center[1] + circle_radius2 * np.sin(angle)
#             circle_points2.append([x, y, circle_center[2]])

#         half_court_bounds = [[(-width / 2), half_length-600, 0], [(width / 2), half_length-600, 0]]

#         half_df = pd.DataFrame(half_court_bounds, columns=['x', 'y', 'z'])
#         circle_df = pd.DataFrame(circle_points, columns=['x', 'y', 'z'])
#         circle_df['line_group'] = 'free_throw_circle'
#         circle_df['color'] = 'court'

#         circle_df2 = pd.DataFrame(circle_points2, columns=['x', 'y', 'z'])
#         circle_df2['line_group'] = 'free_throw_circle'
#         circle_df2['color'] = 'court'

#         half_df['line_group'] = 'half_court'
#         half_df['color'] = 'court'

#         return pd.concat([half_df, circle_df, circle_df2])

#     def __get_backboard_coordinates(self, loc):
#         backboard_start = -self.backboard_width / 2
#         backboard_end = self.backboard_width / 2
#         height = self.backboard_height
#         floor_offset = self.backboard_floor_offset

#         if loc == 'far':
#             offset = self.court_length / 2 - self.backboard_baseline_offset
#         elif loc == 'near':
#             offset = -self.court_length / 2 + self.backboard_baseline_offset

#         backboard_bounds = [
#             [backboard_start, offset, floor_offset],
#             [backboard_start, offset, floor_offset + height],
#             [backboard_end, offset, floor_offset + height],
#             [backboard_end, offset, floor_offset],
#             [backboard_start, offset, floor_offset]
#         ]

#         smaller_rect_width = 1.5*10
#         smaller_rect_height = 1*10
#         hoop_height = self.hoop_loc_z

#         smaller_rect_start_x = backboard_start + (self.backboard_width / 2) - (smaller_rect_width / 2)
#         smaller_rect_end_x = backboard_start + (self.backboard_width / 2) + (smaller_rect_width / 2)
#         smaller_rect_y = offset

#         smaller_rect_bounds = [
#             [smaller_rect_start_x, offset, hoop_height],
#             [smaller_rect_start_x, offset, hoop_height + smaller_rect_height],
#             [smaller_rect_end_x, offset, hoop_height + smaller_rect_height],
#             [smaller_rect_end_x, offset, hoop_height],
#             [smaller_rect_start_x, offset, hoop_height]
#         ]

#         backboard_df = pd.DataFrame(backboard_bounds, columns=['x', 'y', 'z'])
#         backboard_df['line_group'] = f'{loc}_backboard'
#         backboard_df['color'] = 'backboard'

#         smaller_rect_df = pd.DataFrame(smaller_rect_bounds, columns=['x', 'y', 'z'])
#         smaller_rect_df['line_group'] = f'{loc}_smaller_rectangle'
#         smaller_rect_df['color'] = 'backboard'

#         return pd.concat([backboard_df, smaller_rect_df])

#     def __get_three_point_coordinates(self, loc):
#         hoop_loc_x, hoop_loc_y = self.hoop_loc_x, self.hoop_loc_y
#         strt_dst_start = -self.court_width / 2 + self.three_straight_distance
#         strt_dst_end = self.court_width / 2 - self.three_straight_distance
#         strt_len = self.three_straight_length
#         arc_dst = self.three_arc_distance

#         start_straight = [
#             [strt_dst_start, -self.court_length / 2, 0],
#             [strt_dst_start, -self.court_length / 2 + strt_len, 0]
#         ]
#         end_straight = [
#             [strt_dst_end, -self.court_length / 2 + strt_len, 0],
#             [strt_dst_end, -self.court_length / 2, 0]
#         ]
#         line_coordinates = []

#         if loc == 'near':
#             hoop_loc_y = self.court_length / 2 - hoop_loc_y
#             start_straight = [[strt_dst_start, self.court_length / 2, 0], [strt_dst_start, self.court_length / 2 - strt_len, 0]]
#             end_straight = [[strt_dst_end, self.court_length / 2 - strt_len, 0], [strt_dst_end, self.court_length / 2, 0]]

#         a = 1
#         b = -2 * hoop_loc_y
#         d = arc_dst
#         for x_coord in np.linspace(int(strt_dst_start), int(strt_dst_end), 100):
#             c = hoop_loc_y ** 2 + (x_coord - hoop_loc_x) ** 2 - d ** 2

#             y1, y2 = self.calculate_quadratic_values(a, b, c)
#             if loc == 'far':
#                 y_coord = y1
#             if loc == 'near':
#                 y_coord = y2

#             line_coordinates.append([x_coord, y_coord, 0])

#         line_coordinates.extend(end_straight)

#         far_three_df = pd.DataFrame(line_coordinates, columns=['x', 'y', 'z'])
#         far_three_df['line_group'] = f'{loc}_three'
#         far_three_df['color'] = 'court'

#         return far_three_df

#     def __get_hoop_coordinates(self):
#     # Define the number of points to approximate the circle
#         num_points = 100
#         angle_step = 2 * np.pi / num_points

#         # Generate the circle points
#         angles = np.linspace(0, 2 * np.pi, num_points)
#         x = self.hoop_loc_x + self.hoop_radius * np.cos(angles)
#         y = self.hoop_loc_y - self.hoop_radius - 35 + self.hoop_radius * np.sin(angles)  # Adjust the center if needed
#         z = np.full_like(x, 100)  # Assuming a constant z value

#         # Create DataFrame
#         hoop_df = pd.DataFrame({
#             'x': x,
#             'y': y,
#             'z': z
#         })

#         # Add extra columns
#         hoop_df['line_group'] = 'hoop'
#         hoop_df['color'] = 'hoop'
        
#         return hoop_df

#     def get_coordinates(self):
#         court_perimeter_df = self.__get_court_perimeter_coordinates()
#         half_court_df = self.__get_half_court_coordinates()
#         far_backboard_df = self.__get_backboard_coordinates('far')
#         near_backboard_df = self.__get_backboard_coordinates('near')
#         far_three_df = self.__get_three_point_coordinates('far')
#         near_three_df = self.__get_three_point_coordinates('near')
#         hoop_df = self.__get_hoop_coordinates()

#         coordinates = pd.concat([
#             court_perimeter_df,
#             half_court_df,
#             far_backboard_df,
#             # near_backboard_df,
#             # far_three_df,
#             # near_three_df,
#             hoop_df
#         ])
        
#         return coordinates

import pandas as pd
import numpy as np

# TODO: store coordinates in csv file
class CourtCoordinates:
    def __init__(self, year):
        self.hoop_loc_x = 0
        self.hoop_loc_y = 52
        self.hoop_loc_z = 100
        self.court_perimeter_coordinates = []
        self.three_point_line_coordinates = []
        self.backboard_coordinates = []
        self.hoop_coordinates = []
        self.free_throw_line_coordinates = []
        self.court_lines_coordinates_df = pd.DataFrame()
        self.year = year

    @staticmethod
    def calculate_quadratic_values(a, b, c):
        '''
        Given values a, b, and c,
        the function returns the output of the quadratic formula
        '''
        x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

        return x1, x2

    def calculate_court_perimeter_coordinates(self):
        # half court lines
        # x goes from 250 to -250 (50 feet wide)
        # y should go from 0 to 470 (full court is 94 feet long, half court is 47 feet)
        court_perimeter_bounds = [[-250, 0, 0], [250, 0, 0], [250, 450, 0], [-250, 450, 0], [-250, 0, 0]]

        self.court_perimeter_coordinates = court_perimeter_bounds

    def calculate_three_point_line_coordinates(self):
        if int(self.year.split('-')[0]) < 1997:
            d = 220
        else:
            d = 237.5
        # 3point line left side coordinates
        line_coordinates = [[-218, 0, 0], [-218, 82, 0]]
        
        # 3point line arc
        hoop_loc_x, hoop_loc_y = self.hoop_loc_x, self.hoop_loc_y
        a = 1
        b = -2 * 52
        # d = 237.5  # the arc is 23ft and 9inches from the center of the hoop
        for x_coord in range(-218, 218, 2):
            c = hoop_loc_y ** 2 + (hoop_loc_x - x_coord) ** 2 - (d) ** 2
            y_coord = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
            line_coordinates.append([x_coord, y_coord, 0])

        # 3point line right side coordinates
        line_coordinates.append([218, 82, 0])
        line_coordinates.append([218, 0, 0])

        self.three_point_line_coordinates = line_coordinates

    def calculate_backboard_coordinates(self):
        backboard_coordinates = [[30, 40, 90], [30, 40, 130], [-30, 40, 130], [-30, 40, 90], [30, 40, 90]]

        self.backboard_coordinates = backboard_coordinates

    def calculate_hoop_coordinates(self):
        hoop_coordinates_top_half = []
        hoop_coordinates_bottom_half = []

        hoop_center_x, hoop_center_y, hoop_center_z = (self.hoop_loc_x, self.hoop_loc_y, self.hoop_loc_z)
        hoop_min_x, hoop_max_x = (-7.5, 7.5)
        hoop_step = 0.5
        hoop_radius = 7.5

        a = 1
        b = -2 * hoop_center_y
        for hoop_coord_x in np.arange(hoop_min_x, hoop_max_x + hoop_step, hoop_step):
            c = hoop_center_y ** 2 + (hoop_center_x - hoop_coord_x) ** 2 - hoop_radius ** 2
            hoop_coord_y1, hoop_coord_y2 = self.calculate_quadratic_values(a, b, c)

            hoop_coordinates_top_half.append([hoop_coord_x, hoop_coord_y1, hoop_center_z])
            hoop_coordinates_bottom_half.append([hoop_coord_x, hoop_coord_y2, hoop_center_z])

        self.hoop_coordinates = hoop_coordinates_top_half + hoop_coordinates_bottom_half[::-1]
    
    def calculate_free_throw_line_coordinates(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        for i in range(num_points):
            angle = np.pi * i / (num_points - 1)  # Semi-circle from 0 to π
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] + radius * np.sin(angle)
            circle_points.append([x, y, 0])
        
        # Adding lines to the free throw line
        free_throw_line_coordinates = [
            [-90, circle_center[1], 0],  # Left end of the line
            [90, circle_center[1], 0]    # Right end of the line
        ]
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [-90, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-90, baseline_y, 0],
                [90, circle_center[1], 0],   # Right end of the free throw line to baseline
                [90, baseline_y, 0]
            ]
        lines_to_baseline2 = [
                [-75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-75, baseline_y, 0],
                [75, circle_center[1], 0],   # Right end of the free throw line to baseline
                [75, baseline_y, 0]
            ]

    # Combine all coordinates
        self.free_throw_line_coordinates = circle_points + free_throw_line_coordinates 
    
    def calculate_free_throw_line_coordinates2(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [-90, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-90, baseline_y, 0]]
        self.free_throw_line_coordinates = lines_to_baseline
    
    def calculate_free_throw_line_coordinates3(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [90, circle_center[1], 0],   # Right end of the free throw line to baseline
                [90, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline

    def calculate_free_throw_line_coordinates4(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline2 = [
                [-75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-75, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline2
    
    def calculate_free_throw_line_coordinates5(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline2 = [
                [75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [75, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline2
    def __get_hoop_coordinates2(self):
        num_net_lines = 10  # Number of vertical lines in the net
        net_length = 1.75 *10 # Length of the net hanging down from the hoop (in feet)
        initial_radius = 7.5  # Radius at the top of the net

        hoop_net_coordinates = []
        hoop_loc_x, hoop_loc_y, hoop_loc_z = self.hoop_loc_x, self.hoop_loc_y, self.hoop_loc_z


        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines
            
            for j in np.linspace(0, net_length, num=10):
                # Decrease the radius from the initial radius to half of it at the bottom
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                
                x = hoop_loc_x + current_radius * np.cos(angle)
                y = hoop_loc_y + current_radius * np.sin(angle)
                z = hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        
        # Add lines on the other side (negative angles)
        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines + np.pi  # Shift angles to cover the opposite side
            
            for j in np.linspace(0, net_length, num=10):
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                
                x = hoop_loc_x + current_radius * np.cos(angle)
                y = hoop_loc_y + current_radius * np.sin(angle)
                z = hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        self.hoop_net_coordinates = hoop_net_coordinates



    def calculate_court_lines_coordinates(self):
        self.calculate_court_perimeter_coordinates()
        court_df = pd.DataFrame(self.court_perimeter_coordinates, columns=['x', 'y', 'z'])
        court_df['line_id'] = 'outside_perimeter'
        court_df['line_group_id'] = 'court'

        self.calculate_three_point_line_coordinates()
        three_point_line_df = pd.DataFrame(self.three_point_line_coordinates, columns=['x', 'y', 'z'])
        three_point_line_df['line_id'] = 'three_point_line'
        three_point_line_df['line_group_id'] = 'court'

        self.calculate_backboard_coordinates()
        backboard_df = pd.DataFrame(self.backboard_coordinates, columns=['x', 'y', 'z'])
        backboard_df['line_id'] = 'backboard'
        backboard_df['line_group_id'] = 'backboard'
        self.__get_hoop_coordinates2()
        netdf = pd.DataFrame(self.hoop_net_coordinates, columns=['x', 'y', 'z'])
        netdf['line_id'] = 'hoop2'
        netdf['line_group_id'] = 'hoop2'

        self.calculate_hoop_coordinates()
        hoop_df = pd.DataFrame(self.hoop_coordinates, columns=['x', 'y', 'z'])
        hoop_df['line_id'] = 'hoop'
        hoop_df['line_group_id'] = 'hoop'

        self.calculate_free_throw_line_coordinates()
        free_throw_line_df = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df['line_id'] = 'free_throw_line'
        free_throw_line_df['line_group_id'] = 'free_throw_line'

        self.calculate_free_throw_line_coordinates2()
        free_throw_line_df2 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df2['line_id'] = 'free_throw_line2'
        free_throw_line_df2['line_group_id'] = 'free_throw_line2'

        self.calculate_free_throw_line_coordinates3()
        free_throw_line_df3 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df3['line_id'] = 'free_throw_line3'
        free_throw_line_df3['line_group_id'] = 'free_throw_line3'

        self.calculate_free_throw_line_coordinates4()
        free_throw_line_df4 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df4['line_id'] = 'free_throw_line4'
        free_throw_line_df4['line_group_id'] = 'free_throw_line4'

        self.calculate_free_throw_line_coordinates5()
        free_throw_line_df5 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df5['line_id'] = 'free_throw_line5'
        free_throw_line_df5['line_group_id'] = 'free_throw_line5'


        self.court_lines_coordinates_df = pd.concat([
            court_df, three_point_line_df, backboard_df, hoop_df, free_throw_line_df,netdf,free_throw_line_df2,free_throw_line_df3,free_throw_line_df4,free_throw_line_df5
        ], ignore_index=True, axis=0)

    def get_coordinates(self):
        self.calculate_court_lines_coordinates()
        return self.court_lines_coordinates_df
