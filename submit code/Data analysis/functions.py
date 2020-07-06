# the library used
import numpy as np
import pandas as pd
import pysubgroup as ps
import re
from sklearn.feature_extraction import DictVectorizer
import time
import ast
import string
import imblearn
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def tags_preprocess(tags):
    """
    input: tag string
    output: list of individual tags in the given tag string
    function: preprocess a single tag string 
    """
    #tags = tags.replace("'","")
    #tags = tags.replace(" ","")
    #tags = tags.replace("[","")
    #tags = tags.replace("]","")
    tags = str(tags)
    tags = tags.split(" ")
    tags = [x.lower() for x in tags]
    return tags



def ingredients_preprocess(df):
    """
    input: dataframe
    output: list of distinct ingredient list
    function: preprocess the ingredient columns and return a list of distinct ingredients
    """
    distinct_ingredients = []
    dataframe = df
    
    for i in range(len(dataframe)):
        ingredients = dataframe.iloc[i]['ingredient']
        
        r = re.compile('[A-Z]{1}[a-zA-Z]+')
        ingredients = str(ingredients)
        #ingredients = ''.join(i for i in ingredients if not i.isdigit())
        ingredients = ingredients.replace("'","")
        #ingredients = ingredients.replace(" ","")
        ingredients = ingredients.replace("[","")
        ingredients = ingredients.replace("]","")
        # remove text inside parentheses
        ingredients = re.sub(r'\([^())]*\)',"", ingredients)
        ingredients = ingredients.split(" ")
        ingredients = list(filter(r.match, ingredients))
        ingredients = [x.lower() for x in ingredients]
        distinct_ingredients += ingredients
        dataframe.at[i, 'ingredient'] = ingredients
        #dataframe.set_value(i, 'ingredient', ingredients)
        
    return [list(set(distinct_ingredients)),dataframe]



def get_recipe_countries(countries, data):
    """
    input: list of countries, dataframe
    output: selected dataframe whose recipes is from these countries
    function: select the rows in dataframe whose "tag" value contain one country tag
    """
    # add a new column class 
    drop_index = []
    for i in range(len(data)):
        tags = data.loc[i]["tags"]
        tags = tags_preprocess(tags)
        
        country_same =[l for l in countries if l in tags]
            
        if len(country_same) == 1:
            data.at[i, 'label'] = country_same[0]
        if len(country_same) == 0:
            drop_index.append(i)
        if len(country_same) > 1:
            drop_index.append(i)
            #data.at[i, 'label'] = 'overlap'
            
    # drop the columns which has no season tags
    data = data.drop(data.index[drop_index])
    return data


def convert_to_dict(arr):
    """
    Helper function to convect an array of ingredients to a dictionary
    """
    d={}
    for a in arr:
        d[a]=1
    return d


# extract comment user dataset from original dataset
def extract_com_user(data):
    """
    input: dataframe
    output: a new dataframe with all the comment user information
    function: spilt the dictionary of the column 'comment user' in original dataset
    """
    df_com = pd.DataFrame()
    for index, item in data['comment_user'].iteritems():
        if (item != '[]'):
            if (item != 'no comment'):
                array = ast.literal_eval(item)
                df_array = pd.DataFrame(array)
                df_array['recipe_id'] = index
                df_com = pd.concat([df_com,df_array])
    return df_com


def sub_cat_in_com(data):
    """
    input: dataframe
    output: a new dataframe with multi colunms
    function: add one subcategory of recipes to comment user dataset 
    """    
    punct = set(string.punctuation) 
    list_sub_cat = []
    
    df_com2 = pd.DataFrame()
    for index, item in data['calorie_value'].iteritems(): 
        if (item != None):
            list_sub = list(item)
            list_sub = ''.join(x for x in list_sub if x not in punct)
            list_sub_cat.append(list_sub)
    df_sub_cat = pd.DataFrame(list_sub_cat)
    df_sub_cat['calorie_value'] = df_sub_cat
    
    df_sub_cat['recipe_id'] = data['calorie_value'].index     
    df_com2 = pd.concat([df_com2,df_sub_cat])
    return df_com2


def add_recipe_info(data):
    """
    input: dataframe
    output: a new dataframe with multi colunms
    function: add one subcategory of recipes to comment user dataset 
    """    
    punct = set(string.punctuation) 
    # add recipe name in comment users data
    list_recipe_name = []
    df_recipe = pd.DataFrame()
    for index, item in data['recipe_name'].iteritems(): 
        if (item != None):
            list_name = list(item)
            list_name = ''.join(x for x in list_name if x not in punct)
            list_recipe_name.append(list_name)

    df_name = pd.DataFrame(list_recipe_name)
    df_name['recipe_id'] = data['recipe_name'].index 
    df_name = df_name.set_index(["recipe_id"])
    df_name
    df_recipe['recipe_name'] = df_name[0]

    # add recipe difficulty in comment users data
    list_recipe_diff = []
    df_recipe_diff = pd.DataFrame()
    for index, item in data['difficulty'].iteritems(): 
        if (item != None):
            list_diff = list(item)      
            list_diff = ''.join(x for x in list_diff if x not in punct)   
            list_recipe_diff.append(list_diff)

    df_diff = pd.DataFrame(list_recipe_diff)
    df_diff['recipe_id'] = data['difficulty'].index 
    df_diff = df_diff.set_index(["recipe_id"])
    df_recipe['difficulty'] = df_diff[0]
    
    # add recipe avg score in comment users data
    df_recipe_avg = pd.DataFrame()
    list_avg = [i for i in data['avg_score']]

    df_avg = pd.DataFrame(list_avg)
    df_avg['recipe_id'] = data['avg_score'].index
    df_avg = df_avg.set_index(["recipe_id"])
    df_recipe['avg score'] = df_avg[0]

    # add recipe preparation_time in comment users data
    list_recipe_pre = []
    df_recipe_pre = pd.DataFrame()
    for index, item in data['preparation_time'].iteritems(): 
        if (item != None):
            list_pre = list(item)

            list_pre = ''.join(x for x in list_pre if x not in punct)

            list_recipe_pre.append(list_pre)

    df_pre = pd.DataFrame(list_recipe_pre)
    df_pre['recipe_id'] = data['preparation_time'].index 
    df_pre = df_pre.set_index(["recipe_id"])
    df_recipe['preparation_time'] = df_pre[0]

    # add recipe tags in comment users data
    list_recipe_tags = []
    df_recipe_tags = pd.DataFrame()
    for index, item in data['tags'].iteritems(): 
        if (item != None):
            list_tags = list(item)      
            list_tags = ''.join(x for x in list_tags if x not in punct)   
            list_recipe_tags.append(list_tags)


    df_tags = pd.DataFrame(list_recipe_tags)
    df_tags['recipe_id'] = data['tags'].index 
    df_tags = df_tags.set_index(["recipe_id"])
    df_recipe['tags'] = df_tags[0]

    # add recipe ingredient in comment users data
    list_recipe_ingredient = []
    df_recipe_ingredient = pd.DataFrame()
    for index, item in data['ingredient'].iteritems(): 
        if (item != None):
            list_ingredient = list(item) 
            list_ingredient = ''.join(x for x in list_ingredient if x not in punct)
            list_recipe_ingredient.append(list_ingredient)

    df_recipe_ingredient = pd.DataFrame(list_recipe_ingredient)
    df_recipe_ingredient['recipe_id'] = data['ingredient'].index 
    df_recipe_ingredient = df_recipe_ingredient.set_index(["recipe_id"])
    df_recipe['ingredient'] = df_recipe_ingredient[0]
    return df_recipe


def age_group(age):   
    """
    input: age value
    output: group description
    function: divide age value uinto 5 groups 
    """   
    bucket = None
    age = int(age)    
    if age <= 30:
        bucket = '<30 Jahre'     
    if age in range(30, 46):
        bucket = '30-45 Jahre'
    if age in range(45, 61):
        bucket = '45-60 Jahre'
    if age >= 61:
        bucket = '60+ Jahre'
    return bucket


def calorie_level(calorie):   
    """
    input: calorie value
    output: group description
    function: divide calorie value into 3 groups 
    """   
    bucket = None
    calorie = int(calorie)    
    if calorie <= 300:
        bucket = 'low_calorie'    
    if calorie in range(300, 500):
        bucket = 'medium_calorie'        
    if calorie >= 500:
        bucket = 'high_calorie'      
    return bucket


def remove_None(data, name):
    """
    Helper function to remove None value in one column
    """ 
    y = data[data[name] == 'None']
    index_n = y.index.tolist()
    data = data.drop(index = index_n)
    return data


def add_target(data, calorie_level):
    df_sub_group[country] = df_dum_car[calorie_level]
    return df_sub_group


def pre_time_group(pre_time):   
    pre_time = int(pre_time)    
    if pre_time <= 20:
        bucket = '<20 Min'    
    if pre_time in range(20, 31):
        bucket = '20-30 Min'                
    if pre_time >= 31:
        bucket = '30+ Min'
    return bucket

def tags_preprocessing(tags):
    tags = tags.replace("'","")
    tags = tags.replace(" ","")
    tags = tags.replace("[","")
    tags = tags.replace("]","")
    tags = tags.split(",")
    tags = [x.lower() for x in tags]
    return tags

def calculate_mean(data_tags,tag_name):
    """
    Helper function to calculate the mean of average score 
    of recipes with special tags
    """ 
    
    score_list = []

    for i in range(len(data_tags)):
        tags = data_tags.loc[i]['tags']
        tags = tags_preprocessing(tags)        
        if tag_name in tags:
            score_list.append(data_tags.at[i,'avg_score'])
    df_tags = pd.DataFrame(score_list)
    df_tags = df_tags.rename(columns={0:tag_name})
    m = df_tags.mean()
    s = df_tags.sum()
    return m

def get_cooking_method(cook_method, data):

    # add a new column class 
    drop_index = []
    for i in range(len(data)):
        tags = data.loc[i]["tags"]
        tags = tags_preprocessing(tags)
        
        cooking_same =[l for l in cook_method if l in tags]
            
        if len(cooking_same) == 1:
            data.at[i, 'cooking method'] = cooking_same[0]
        if len(cooking_same) == 0:
            drop_index.append(i)
        if len(cooking_same) > 1:
            drop_index.append(i)
            
    # drop the columns which has no cooking method tags
    data = data.drop(data.index[drop_index])
    return data

