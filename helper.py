import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class RGB:
    '''RGB class to handle colour feature'''
    def __init__(self,r,g,b):
        self.r = r
        self.b = b
        self.g = g
#colour_dist_red function get the colour distance of input colour with red. 
#Colour distance to represent how different/similar two colours are
def colour_dist_red(e1):
    e2 = RGB(255,0,0)
    rmean = (e1.r + e2.r ) // 2
    r = int(e1.r - e2.r)
    g = int(e1.g - e2.g)
    b = int(e1.b - e2.b)
    return math.sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))/764.84

def colour_dist_green(e1):
    e2 = RGB(0,255,0)
    rmean = (e1.r + e2.r ) // 2
    r = int(e1.r - e2.r)
    g = int(e1.g - e2.g)
    b = int(e1.b - e2.b)
    return math.sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))/764.84

def colour_dist_blue(e1):
    e2 = RGB(0,0,255)
    rmean = (e1.r + e2.r ) // 2
    r = int(e1.r - e2.r)
    g = int(e1.g - e2.g)
    b = int(e1.b - e2.b)
    return math.sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))/764.84

def getFaceGlassShapeRatings(fileName):
	df_FS_Rating = pd.read_csv(fileName)
	return df_FS_Rating
def preProcessData(faceShape, FGR_fileName, model_fileName):
	df_shop = pd.read_csv(model_fileName)
	df_FS_Rating = getFaceGlassShapeRatings(FGR_fileName)

	print(df_FS_Rating)
	colour_dist_list =[]
	colour = df_shop["Colour"]
	for c in colour:
	    c_list = list(map(int,c.split(",")))
	    #print(c_list)
	    colour_list =[]
	    colour_RGB = RGB(c_list[0],c_list[1],c_list[2])
	    colour_list.append(round(1-colour_dist_red(colour_RGB),5))
	    colour_list.append(round(1-colour_dist_green(colour_RGB),5))
	    colour_list.append(round(1-colour_dist_blue(colour_RGB),5))
	    colour_dist_list.append(colour_list)

	df_RGBSim_poly = pd.DataFrame(colour_dist_list,columns=["sim_R","sim_G","sim_B"])
	df_RGBSim_poly = pd.concat([df_shop[["Name", "Shape"]],df_RGBSim_poly],axis = 1)
	df_RGBSim_poly["shapePoint"] = 0

	glassShape = ["round", "cat", "oval", "rectangular", "aviator", "wrap"]

	for shape in glassShape:
	    df_RGBSim_poly.loc[df_RGBSim_poly["Shape"] == shape, "shapePoint"] = df_FS_Rating.loc[df_FS_Rating["face_shape"] == faceShape, shape].values[0]

	return df_RGBSim_poly
def predictValue(df_RGBSim_poly, ratings):
    df_RGBSim_poly['rating'] = None
    for i in range (1, 8): #suppose n_models = 8
        keyModel = "M" + str(i)
        if keyModel in ratings:
            df_RGBSim_poly.loc[df_RGBSim_poly['Name'] == keyModel, 'rating'] = int(ratings[keyModel])

    #use all NA to train model
    temp = df_RGBSim_poly.dropna()
    features_train = temp[["sim_R", "sim_G", "sim_B", "shapePoint"]]
    rating_train = temp["rating"]

    poly_reg = PolynomialFeatures(degree=9)
    features_train_poly = poly_reg.fit_transform(features_train)
    pol_reg = LinearRegression()
    poly_model = pol_reg.fit(features_train_poly,rating_train)

    score = poly_model.score(features_train_poly, rating_train)
    print(f'Score is {score}')

    rating_train_pred = pol_reg.predict(poly_reg.fit_transform(df_RGBSim_poly[["sim_R", "sim_G", "sim_B", "shapePoint"]]))
    return rating_train_pred

