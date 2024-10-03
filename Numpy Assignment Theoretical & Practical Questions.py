#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#                             Theoretical Questions:


# 1. Explain the purpose and advantages of NumPy in scientific computing and data analysis. How does it
# enhance Python's capabilities for numerical operations?


# 2. Compare and contrast np.mean() and np.average() functions in NumPy. When would you use one over the
# other?


# 3. Describe the methods for reversing a NumPy array along different axes. Provide examples for 1D and 2D
# arrays.


# 4. How can you determine the data type of elements in a NumPy array? Discuss the importance of data types
# in memory management and performance.


# 5. Define ndarrays in NumPy and explain their key features. How do they differ from standard Python lists?


# 6. Analyze the performance benefits of NumPy arrays over Python lists for large-scale numerical operations.


# 7. Compare vstack() and hstack() functions in NumPy. Provide examples demonstrating their usage and
# output.


# 8. Explain the differences between fliplr() and flipud() methods in NumPy, including their effects on various
# array dimensions.


# 9. Discuss the functionality of the array_split() method in NumPy. How does it handle uneven splits?


# 10. Explain the concepts of vectorization and broadcasting in NumPy. How do they contribute to efficient array
# operations?


# In[ ]:


# Practical Questions:
        
        
# 1. Create a 3x3 NumPy array with random integers between 1 and 100. Then, interchange its rows and columns.


# 2. Generate a 1D NumPy array with 10 elements. Reshape it into a 2x5 array, then into a 5x2 array.


# 3. Create a 4x4 NumPy array with random float values. Add a border of zeros around it, resulting in a 6x6 array.


# 4. Using NumPy, create an array of integers from 10 to 60 with a step of 5.


# 5. Create a NumPy array of strings ['python', 'numpy', 'pandas']. Apply different case transformations
# (uppercase, lowercase, title case, etc.) to each element.


# 6. Generate a NumPy array of words. Insert a space between each character of every word in the array.


# 7. Create two 2D NumPy arrays and perform element-wise addition, subtraction, multiplication, and division.


# 8. Use NumPy to create a 5x5 identity matrix, then extract its diagonal elements.


# 9. Generate a NumPy array of 100 random integers between 0 and 1000. Find and display all prime numbers in
# this array.


# 10. Create a NumPy array representing daily temperatures for a month. Calculate and display the weekly
# averages.


# In[ ]:


# Theoretical Questions:
    


# In[ ]:


#  Ques 1

# Explain the purpose and advantages of NumPy in scientific computing and data analysis. How does it enhance Python's capabilities for numerical operations?


# Purpose of NumPy in Scientific Computing and Data Analysis

# NumPy (Numerical Python) is a fundamental library in Python used for numerical and scientific computing.
# Its primary purpose is to provide a powerful N-dimensional array object and functions for efficiently
# performing numerical operations on these arrays. In scientific computing and data analysis, 
# where handling large datasets, performing matrix manipulations, and carrying out complex mathematical
# operations is common, NumPy offers the following advantages:

# Key Advantages of NumPy:
    
# Efficient Array Operations:

# NumPy introduces the ndarray (N-dimensional array), which allows for efficient storage and manipulation 
# of large datasets. Unlike Python lists, NumPy arrays are optimized for performance, enabling faster computations.

# Vectorization:
    
# Operations on NumPy arrays are performed element-wise without the need for explicit loops. 
# This is called vectorization, which makes calculations faster and the code cleaner.


# Memory Efficiency:

# NumPy arrays are more memory-efficient than Python lists because they store elements of the same 
# data type in contiguous memory blocks, reducing overhead and improving speed.

# Mathematical Functions:

# NumPy includes a wide range of mathematical functions like linear algebra, random number generation, 
# and statistical operations. This eliminates the need for complex and time-consuming implementations.

# Support for Multidimensional Arrays (Matrices):

# While Python lists are typically one-dimensional, NumPy supports multi-dimensional arrays (e.g., matrices and tensors),
# which are commonly used in scientific and machine learning applications.

# Broadcasting:

# NumPy supports broadcasting, allowing you to perform operations on arrays of different shapes. 
# This feature simplifies operations and makes them more intuitive.


# In[ ]:


# How NumPy Enhances Python’s Capabilities for Numerical Operations:
    
# Performance:

# NumPy is much faster than Python's built-in list for numerical operations because it uses optimized C code under the hood. 
# It handles large data efficiently and is capable of performing complex computations faster than pure Python code.

# Ease of Use:

# With NumPy, operations that require manual loops in standard Python can be replaced with single, concise commands. 
# This simplifies the code and makes it more readable and maintainable.

# Handling of Multidimensional Data:

# Unlike Python lists, which are limited to 1-dimensional data, NumPy arrays can be easily used to represent and 
# manipulate multi-dimensional data, such as 2D matrices and higher-dimensional tensors, which are essential in scientific computing.

# Numerical Stability and Precision:

# NumPy functions are designed to provide accurate and stable results for numerical operations. It also supports a 
# wide range of data types (integers, floats, etc.), allowing for more precise calculations.

# Efficient Mathematical Computations:

# NumPy’s extensive library of mathematical functions makes it easy to perform tasks such as matrix multiplication,
# Fourier transforms, and linear algebra operations, all of which are critical in scientific computing and data analysis.


# In[ ]:


# Ques 2 

# Compare and contrast np.mean() and np.average() functions in NumPy. When would you use one over the
# other?

# Both np.mean() and np.average() are functions in NumPy that are used to calculate the average value of elements in an array, but they have some key differences in how they work, particularly when dealing with weights.

# 1. np.mean():

# Purpose: It calculates the arithmetic mean (or simple average) of the array elements.

# Usage: np.mean() is used when you want to compute the average of the values without considering any weights.

# Parameters:

# array: Input array or sequence.

# axis: (Optional) Specifies the axis along which to compute the mean. If not specified, it computes the mean of all elements.

# dtype: (Optional) Determines the type of the output (e.g., integer or float).

# keepdims: (Optional) If set to True, the result will retain the dimensions of the input array.


# In[ ]:


# 2. np.average():

# Purpose: It calculates the weighted average of array elements. If no weights are provided, it defaults to calculating the same result as np.mean().

# Usage: np.average() is used when you want to take into account the weights of each element. You can assign a higher weight to certain elements to reflect their importance in the calculation.

# Parameters:

# array: Input array or sequence.

# weights: (Optional) An array of the same shape as the input, containing the weights associated with each element.

# axis: (Optional) Specifies the axis along which to compute the average. If not specified, it computes the average of all elements.

# returned: (Optional) If set to True, the function will return a tuple containing the average and the sum of the weights.


# In[ ]:


# When to Use Each:

# Use np.mean():

# When you want a simple, unweighted average of elements.
# When all data points should have equal importance in your computation.

# Use np.average():

# When the data points have different levels of importance and you need to assign weights.
# In scenarios like statistical models, financial calculations, or any application where certain data points contribute more than others.


# In[ ]:


# Ques 3

# Describe the methods for reversing a NumPy array along different axes. Provide examples for 1D and 2D
# arrays.

# In NumPy, there are several methods to reverse an array along different axes, whether it's a 1D array or a 2D array. Below are some of the key methods you can use, with examples for both 1D and 2D arrays.

# Methods for Reversing a NumPy Array:

# 1. Using Slicing ([::-1]):

# This is the simplest and most common method for reversing arrays in NumPy. You can reverse an array along a given axis by using Python’s slicing technique (start:stop:step) where a step of -1 reverses the elements.

# For 1D Arrays:


# In[1]:


import numpy as np

arr_1d = np.array([1, 2, 3, 4, 5])
reversed_1d = arr_1d[::-1]
print(reversed_1d)  # Output: [5 4 3 2 1]


# In[2]:


# For 2D Arrays (Row-wise or Column-wise):

# To reverse along rows:
    
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
reversed_rows = arr_2d[::-1, :]  # Reverses the rows
print(reversed_rows)
# Output:
# [[7 8 9]
#  [4 5 6]
#  [1 2 3]]


# In[4]:


# To reverse along columns:
reversed_columns = arr_2d[:, ::-1]  # Reverses the columns
print(reversed_columns)
# Output:
# [[3 2 1]
#  [6 5 4]
#  [9 8 7]]


# In[ ]:


# Summary of Methods:

#  Method	         Reverses	                         Description 
 
# [::-1]   	     Array along any axis	             Uses Python slicing to reverse along an axis.
# np.flip()        Array along specified axis	         General method to flip array along specified axis
# np.fliplr()	     2D array along columns	             Shorthand for flipping left-right (columns).
# np.flipud()	     2D array along rows	             Shorthand for flipping up-down (rows).
# np.transpose()   Swaps axes, combined with slicing	 Allows swapping of axes and reversing.


# In[ ]:


# Ques 4 

# How can you determine the data type of elements in a NumPy array? Discuss the importance of data types
# in memory management and performance.

# Determining the Data Type of Elements in a NumPy Array

# In NumPy, you can determine the data type of the elements in an array using the dtype attribute.
# The dtype (data type) tells you what type of elements the array contains, such as integers (int), 
# floating-point numbers (float), booleans (bool), etc.

# Importance of Data Types in Memory Management and Performance

# Memory Efficiency:

# NumPy arrays are more memory-efficient compared to Python lists because they store elements of the same data type. The choice of data type impacts how much memory each element occupies.
# For example, int8 uses 1 byte per element, while int64 uses 8 bytes. By carefully selecting the appropriate data type (e.g., int32 instead of int64 if the range of numbers is smaller), you can save a lot of memory when dealing with large datasets.

# Performance:

# NumPy operations are highly optimized, and the data type (dtype) of an array affects the performance of numerical computations.

# For example, operations on arrays of float32 or int32 types are generally faster than those on float64 or int64 because smaller data types require less processing power and memory bandwidth.


# In[ ]:


# Ques 5

# Define ndarrays in NumPy and explain their key features. How do they differ from standard Python lists?


# Definition of ndarray in NumPy

# An ndarray (N-dimensional array) is the fundamental data structure in NumPy, used for storing homogeneous, multidimensional data. It is a grid of values, all of the same data type, and indexed by a tuple of non-negative integers. The number of dimensions is called the array's rank, and the shape is the tuple of integers representing the size along each dimension.

# Key Features of ndarray:

# Homogeneous Elements:

# All elements in an ndarray must be of the same data type (e.g., all integers or all floats), which allows for optimized memory usage and faster computation.

# Multidimensional:

# ndarray supports arrays of any number of dimensions (1D, 2D, 3D, and higher). For example, a 1D array is a list of numbers, while a 2D array is a matrix, and 3D arrays can represent tensors.

# Efficient Memory Usage:

# NumPy arrays are stored in contiguous blocks of memory, unlike Python lists, which are arrays of pointers to objects. This improves memory access speed and efficiency.

# Vectorized Operations:

# Operations on ndarray objects are performed element-wise and are vectorized, meaning you can apply mathematical operations to entire arrays without writing explicit loops.


# In[ ]:


# Differences Between ndarray and Python Lists
# Homogeneity:

# ndarray: All elements must be of the same data type (e.g., all integers or all floats).
# Python List: Can contain elements of different data types (e.g., integers, floats, strings).

# Multidimensional Support:

# ndarray: Supports multiple dimensions (1D, 2D, 3D, etc.).
# Python List: Lists are typically one-dimensional, though you can create lists of lists to simulate multidimensional arrays, but they are less efficient.

# Memory Efficiency:

# ndarray: Uses contiguous memory blocks for efficient data storage and access, leading to better performance with large datasets.
# Python List: Stores references to objects, leading to higher memory overhead and slower access times for large data structures.

# Vectorized Operations:

# ndarray: Supports vectorized operations, where mathematical operations can be applied to entire arrays at once.
# Python List: Requires loops for element-wise operations; doesn't support native vectorization.

# Performance:

# ndarray: Highly optimized for numerical computations, making it much faster for large datasets or mathematical operations.
# Python List: Slower for numerical computations due to its flexibility and lack of optimization for homogeneous data.


# In[ ]:


# Ques 6

# Analyze the performance benefits of NumPy arrays over Python lists for large-scale numerical operations.

# Performance Benefits of NumPy Arrays Over Python Lists for Large-Scale Numerical Operations
# When it comes to large-scale numerical operations, NumPy arrays (ndarray) offer significant advantages over standard Python lists in terms of memory efficiency, speed, and functionality. Here’s a breakdown of these benefits:


# In[ ]:


# NumPy arrays provide significant performance benefits over Python lists when dealing with large-scale numerical operations, thanks to:

# 1 Memory efficiency through homogeneous data types and contiguous memory blocks.

# 2 Faster execution due to vectorized operations, broadcasting, and optimized mathematical functions.

# 3 Advanced support for multidimensional arrays and numerical operations like matrix multiplication.

# 4 Parallel processing capabilities that make operations on large datasets even faster.

# 5 Efficient slicing and indexing for easy manipulation of large arrays.

# When dealing with large datasets or heavy numerical computations, NumPy arrays are the preferred choice for speed and memory optimization.


# In[ ]:


# Ques 7

# Compare vstack() and hstack() functions in NumPy. Provide examples demonstrating their usage and
# output.


# In NumPy, vstack() and hstack() are two functions used for stacking arrays along different axes. Let's compare them and see how they work with examples.

# 1. vstack() – Vertical Stacking

# vstack() stacks arrays vertically, i.e., along rows (axis=0).
# It combines multiple arrays into a single array by stacking them one on top of the other.
# Arrays must have the same number of columns for vstack() to work.


# Example:


# In[1]:


import numpy as np

# Create two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack them vertically
result = np.vstack((arr1, arr2))

print(result)


# In[ ]:


# 2. hstack() – Horizontal Stacking

# hstack() stacks arrays horizontally, i.e., along columns (axis=1).

# It combines multiple arrays side by side, stacking them one after the other horizontally.

# Arrays must have the same number of rows for hstack() to work.

# Example


# In[2]:


# Create two 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Stack them horizontally
result = np.hstack((arr1, arr2))

print(result)


# In[ ]:


# Summary:
    
# Use vstack() when you want to combine arrays vertically (one on top of another).

# Use hstack() when you want to combine arrays horizontally (side by side).


# In[ ]:


# Ques 8

# Explain the differences between fliplr() and flipud() methods in NumPy, including their effects on various
# array dimensions.


# In NumPy, fliplr() and flipud() are two functions that allow you to flip or reverse the elements of an array along different axes. Here’s a breakdown of their differences and how they affect arrays of various dimensions.

# 1. fliplr() – Flip Left to Right

# fliplr() flips the array horizontally, i.e., from left to right.
# It is only applicable to arrays with 2 or more dimensions (i.e., it works on the last axis, axis=1).
# It reverses the order of elements in each row while keeping the rows in their original order.

# Example:


# In[3]:


import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Flip the array left to right
result = np.fliplr(arr)

print(result)


# In[ ]:


# 2. flipud() – Flip Up to Down

# flipud() flips the array vertically, i.e., from up to down.

# It reverses the order of the rows but keeps the columns in the original order.

# Unlike fliplr(), flipud() can be applied to both 1D and 2D arrays, as it works along axis=0.

# Example with a 2D array:


# In[4]:


# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Flip the array up to down
result = np.flipud(arr)

print(result)


# In[ ]:


# Summary:
    
# fliplr() flips the array horizontally, reversing the order of elements in each row.
# flipud() flips the array vertically, reversing the order of the rows.
# flipud() can handle 1D arrays, while fliplr() requires at least a 2D array.


# In[ ]:


# Ques 9

# Discuss the functionality of the array_split() method in NumPy. How does it handle uneven splits?

# The array_split() method in NumPy is used to split an array into multiple sub-arrays. It is quite similar to the split() function but with the added flexibility of handling uneven splits.

# Key Features of array_split():
    
# Basic Functionality: It splits an array into n sub-arrays along a specified axis.
# Uneven Splits: If the array cannot be divided evenly into the desired number of sub-arrays, array_split() ensures that the resulting sub-arrays are of unequal size, with the first sub-arrays being slightly larger.


# In[ ]:


# Syntax:
numpy.array_split(ary, indices_or_sections, axis=0)

# ary: The input array you want to split.
# indices_or_sections: The number of sub-arrays or the indices at which to split the array.
# axis: The axis along which to split the array (default is axis=0, i.e., row-wise for 2D arrays).


# In[ ]:


# Handling Uneven Splits:

# When the array cannot be evenly divided into sub-arrays, array_split() allocates extra elements to some of the sub-arrays. Specifically:

# The first few sub-arrays will contain one extra element, while the remaining ones will have fewer elements.
# This ensures that all the data in the array is still represented, even when the split is uneven.


# In[6]:


# Example 1: Even Split

import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6])

# Split the array into 3 equal parts
result = np.array_split(arr, 3)

print(result)


# In[7]:


# Example 2: Uneven Split

arr = np.array([1, 2, 3, 4, 5])

# Split the array into 3 parts (uneven)
result = np.array_split(arr, 3)

print(result)


# In[ ]:


# Ques 10

# Explain the concepts of vectorization and broadcasting in NumPy. How do they contribute to efficient array
# operations?


# Vectorization and broadcasting are two core concepts in NumPy that enable efficient and optimized array
# operations. They play a crucial role in making computations faster by leveraging low-level optimizations 
# without needing explicit Python loops.

# 1. Vectorization in NumPy

# Vectorization refers to the ability to perform operations on entire arrays (or vectors) at once, 
# without writing explicit loops. It allows you to apply arithmetic or mathematical operations 
# element-wise on entire arrays in one go.

# Key Benefits:

# Performance: Vectorized operations are much faster than equivalent operations using explicit loops in Python. 
# This is because NumPy’s vectorized functions are implemented in C and take advantage of optimized,
# low-level operations.

# Cleaner Code: Vectorization removes the need for manually writing loops, leading to more readable 
# and concise code.

# Example of Vectorization:
    
# Instead of looping through elements one by one, you can perform operations directly on arrays.


# In[8]:


import numpy as np

# Two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vectorized addition
result = a + b  # Element-wise addition

print(result)


# In[ ]:


# 2. Broadcasting in NumPy

# Broadcasting is a mechanism that allows NumPy to perform element-wise operations on arrays with different shapes, without making copies of the data. It stretches the smaller array across the larger one, as if it were of the same shape, but without creating unnecessary memory overhead.

# Rules for Broadcasting:

# If two arrays have different dimensions, NumPy pads the smaller array’s shape with ones (on the left) until both arrays have the same number of dimensions.
# If the sizes of the dimensions of the two arrays match or one of the dimensions is 1, the arrays are compatible and broadcasting can occur.
# If any dimension sizes differ and none of them is 1, broadcasting cannot happen, and an error is raised.

# Example of Broadcasting:


# In[9]:


import numpy as np

# Array of shape (3,)
a = np.array([1, 2, 3])

# Scalar (0D array)
b = 2

# Broadcasting the scalar over the array
result = a * b  # Multiplies each element of `a` by `b`

print(result)


# In[ ]:


# Summary: How Vectorization and Broadcasting Contribute to Efficient Array Operations
    
# Vectorization speeds up numerical computations by eliminating explicit loops and performing operations on entire arrays at once, leveraging highly optimized C code.

# Broadcasting allows NumPy to efficiently handle arrays of different shapes in element-wise operations without creating unnecessary intermediate arrays or loops.

# Both of these features contribute to making NumPy an efficient tool for numerical computing and large-scale data analysis. They allow operations to be performed faster and with less memory overhead compared to traditional Python loops.


# In[ ]:


# Practical Questions:
    


# In[10]:


# Ques 1

# Create a 3x3 NumPy array with random integers between 1 and 100. Then, interchange its rows and columns.


import numpy as np

# Create a 3x3 NumPy array with random integers between 1 and 100
array = np.random.randint(1, 101, (3, 3))

print("Original array:")
print(array)

# Interchange rows and columns
interchanged_array = array.T

print("Interchanged array:")
print(interchanged_array)


# In[16]:


# Ques 2

# Generate a 1D NumPy array with 10 elements. Reshape it into a 2x5 array, then into a 5x2 array.

# Generate a 1D NumPy array with 10 elements
array_1d = np.arange(10)

# Reshape it into a 2x5 array
array_2x5 = array_1d.reshape(2, 5)

# Reshape it into a 5x2 array
array_5x2 = array_2x5.reshape(5, 2)

array_1d, array_2x5, array_5x2


# In[19]:


# Ques 3

# Create a 4x4 NumPy array with random float values. Add a border of zeros around it, resulting in a 6x6 array.

# Creating a 4x4 NumPy array with random float values
array_4x4 = np.random.rand(4, 4)

# Adding a border of zeros around it to make it a 6x6 array
array_6x6 = np.pad(array_4x4, pad_width=1, mode='constant', constant_values=0)

array_4x4, array_6x6


# In[21]:


# Ques 4

# Using NumPy, create an array of integers from 10 to 60 with a step of 5.

# Creating an array of integers from 10 to 60 with a step of 5
array_integers = np.arange(10, 61, 5)
array_integers


# In[22]:


# Ques 5

# Create a NumPy array of strings ['python', 'numpy', 'pandas']. Apply different case transformations
# (uppercase, lowercase, title case, etc.) to each element.

# Creating a NumPy array of strings
string_array = np.array(['python', 'numpy', 'pandas'])

# Applying different case transformations
uppercase_array = np.char.upper(string_array)  # Uppercase
lowercase_array = np.char.lower(string_array)  # Lowercase
titlecase_array = np.char.title(string_array)   # Title case
capitalize_array = np.char.capitalize(string_array)  # Capitalize

uppercase_array, lowercase_array, titlecase_array, capitalize_array


# In[23]:


# Ques 6

# Generate a NumPy array of words. Insert a space between each character of every word in the array.

# Creating a NumPy array of words
words_array = np.array(['hello', 'world', 'numpy', 'python'])

# Inserting a space between each character of every word
spaced_words_array = np.char.join(' ', words_array)

spaced_words_array


# In[24]:


# Ques 7 

# Create two 2D NumPy arrays and perform element-wise addition, subtraction, multiplication, and division.

# Creating two 2D NumPy arrays
array_1 = np.array([[1, 2, 3], [4, 5, 6]])
array_2 = np.array([[7, 8, 9], [10, 11, 12]])

# Performing element-wise addition
addition_result = array_1 + array_2

# Performing element-wise subtraction
subtraction_result = array_1 - array_2

# Performing element-wise multiplication
multiplication_result = array_1 * array_2

# Performing element-wise division
division_result = array_1 / array_2

addition_result, subtraction_result, multiplication_result, division_result


# In[26]:


# Ques 8

# Use NumPy to create a 5x5 identity matrix, then extract its diagonal elements.

# Creating a 5x5 identity matrix
identity_matrix = np.eye(5)

# Extracting the diagonal elements
diagonal_elements = np.diagonal(identity_matrix)

identity_matrix, diagonal_elements


# In[33]:


# Ques 9

# Generate a NumPy array of 100 random integers between 0 and 1000. Find and display all prime numbers in
# this array.

import numpy as np

def is_prime(num):
    """Checks if a number is prime."""
    if num <= 1:
        return False
    if num <= 3:
        return True
    if num % 2 == 0 or num % 3 == 0:
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    return True

# Generate a NumPy array of 100 random integers between 0 and 1000
random_array = np.random.randint(0, 1001, 100)

# Find prime numbers in the array
prime_numbers = [num for num in random_array if is_prime(num)]

# Display the generated random array and the prime numbers
print("Random Array:", random_array)
print("Prime numbers in the array:")
print(prime_numbers)


# In[36]:


# Ques 10

# Create a NumPy array representing daily temperatures for a month. Calculate and display the weekly
# averages.

import numpy as np

# Generate a NumPy array representing daily temperatures for 30 days (for a month)
# Here, we will use random temperatures between 20 and 35 degrees Celsius
daily_temperatures = np.random.randint(20, 36, size=30)

# Print the daily temperatures for reference
print("Daily Temperatures for the Month:")
print(daily_temperatures)

# Calculate the weekly averages
# Reshape the daily temperatures into a 4-week format (4 rows for 4 weeks and 7 columns for days)
# We take the first 28 days to fit into 4 weeks (4*7 = 28)
weekly_temperatures = daily_temperatures[:28].reshape(4, 7)

# Calculate the average for each week
weekly_averages = np.mean(weekly_temperatures, axis=1)

# Display the weekly averages
print("\nWeekly Averages:")
for week_number, average in enumerate(weekly_averages, start=1):
    print(f"Week {week_number}: {average:.2f}°C")


# In[ ]:





# In[ ]:




