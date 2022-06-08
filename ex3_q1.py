import time
import numpy
import numpy as np
import scipy.spatial
from numpy import genfromtxt, int64
from random import sample

chain_length = 100
num_of_examples0 = 1
num_of_examples01 = 2
num_of_examples012 = 3
num_of_examples0123 = 4
num_of_examples01234 = 5
num_of_examples012345 = 6
num_of_examples0123456 = 7
num_of_examples01234567 = 8
num_of_examples012345678 = 9
num_of_examples0123456789 = 10

# go through 10 lines which each line represent digit
my_data0 = genfromtxt("0_x1_line_by_line.txt", delimiter=' ')
my_data01 = genfromtxt("01_x1_line_by_line.txt", delimiter=' ')
my_data012 = genfromtxt("012_x1_line_by_line.txt", delimiter=' ')
my_data0123 = genfromtxt("0123_x1_line_by_line.txt", delimiter=' ')
my_data01234 = genfromtxt("01234_x1_line_by_line.txt", delimiter=' ')
my_data012345 = genfromtxt("012345_x1_line_by_line.txt", delimiter=' ')
my_data0123456 = genfromtxt("0123456_x1_line_by_line.txt", delimiter=' ')
my_data01234567 = genfromtxt("01234567_x1_line_by_line.txt", delimiter=' ')
my_data012345678 = genfromtxt("012345678_x1_line_by_line.txt", delimiter=' ')
my_data0123456789 = genfromtxt("0123456789_x1_line_by_line.txt", delimiter=' ')

def iniazlize_T_matrix(data,_weight_matrix, _num_of_examples):
    if data.ndim != 1:
        for i in range(chain_length-1):
            for j in range(i+1, chain_length):
                counter = 0
                for bit in range(_num_of_examples):
                    if data[:, i][bit] == data[:, j][bit]:
                        counter += 1
                    else:
                        counter -= 1
                _weight_matrix[i][j] = counter
                _weight_matrix[j][i] = counter
    # else one Dim
    else:
        for i in range(chain_length-1):
            for j in range(i+1, chain_length):
                counter = 0
                if data[i]== data[j]:
                        counter += 1
                else:
                    counter -= 1
                _weight_matrix[i][j] = counter
                _weight_matrix[j][i] = counter
    return _weight_matrix


class CreateHopfieldNetwork:
    def __init__(self, _data,_num_of_examples):
        self.data = _data
        self.weight = numpy.zeros((chain_length, chain_length))
        self.weight = iniazlize_T_matrix(_data, self.weight, _num_of_examples)


def randomize_line(_data):
    data_with_changes = numpy.copy(_data)
    num_of_changes = int(0.1*chain_length)
    if _data.ndim != 1:
        for line in data_with_changes:
            i_index = sample(list(range(chain_length)), k=num_of_changes) # select k samples without replacement
            for i in i_index:
                # convert from float to int because i,j it is indexes
                i = int(i)
                if line[i] == 0:
                    line[i] = 1
                else:
                    line[i] = 0
    else:
        i_index = sample(list(range(chain_length)), k=num_of_changes)  # select k samples without replacement
        for i in i_index:
            # convert from float to int because i,j it is indexes
            i = int(i)
            if data_with_changes[i] == 0:
                data_with_changes[i] = 1
            else:
                data_with_changes[i] = 0
    return data_with_changes


def randomize_data(_samples):
    # randomize
    # print(_samples.ndim)
    # print(np.shape(_samples)[0])
    x = _samples.shape[0]
    if _samples.ndim != 1:
        after_randomize_all_samples = numpy.zeros(shape=(x, chain_length))
    else:
        after_randomize_all_samples = numpy.zeros(shape=(1, chain_length))
    if _samples.ndim != 1:
        for index in range(x):
            res = randomize_line(_samples[index])
            after_randomize_all_samples[index] = np.copy(res)

    else:
        for index in range(1):
            res = randomize_line(_samples)
            after_randomize_all_samples[index] = np.copy(res)

    return after_randomize_all_samples


def check_me(_network, _data, line):
    result = numpy.copy(line)
    x = line.shape[0]
    random_index = sample(list(range(chain_length)), k=chain_length)
    flag_convergence = False
    while not flag_convergence:
        # in order to prevent share memory
        one_before_result = numpy.copy(result)
        for i in range(x):
            # digit is like  * from lecture 9 page 49
            for digit in random_index:
                sum = 0
                for digit2 in range(chain_length):
                    if digit != digit2:
                        sum += one_before_result[i][digit2]*_network.weight[digit][digit2]
                if sum >= 0:
                    result[i][digit] = 1
                else:
                    result[i][digit] = 0
            counter = 0
            for index in range(chain_length):
                if result[i][index] == one_before_result[i][index]:
                    counter += 1
            if counter == chain_length:
                flag_convergence = True
    return result

def check_if_succses(original_data, result):
    counter = 0
    hamming_sum = 0
    x = result.shape[0]
    if (x==1):
        hamming_dis = scipy.spatial.distance.hamming(original_data, result) * 100
        hamming_sum = hamming_sum + hamming_dis
    else:
        for i in range(x):
            hamming_dis = scipy.spatial.distance.hamming(original_data[i], result[i]) * 100
            hamming_sum = hamming_sum + hamming_dis
    percent = ((100*x)-(hamming_sum)) / (100*x)*100
    # percent = (percent)
    return round(percent)

def percent_check(total_percent):
    if (int(total_percent) >= 90):
        return "success"
    else:
        return "fail"

if __name__ == '__main__':
    print("Creates Hopfield Network for each input file...\n")
    time.sleep(1)
    print("Print results:\n")
    time.sleep(1)
    # 0
    hopf_network_0 = CreateHopfieldNetwork(my_data0, num_of_examples0)
    data_after_changes0 = randomize_data(my_data0)
    result = check_me(hopf_network_0, my_data0, data_after_changes0)
    percent = check_if_succses(my_data0, result)
    print("The network has " + str(percent) + "% success to remember model 0. It's a "+percent_check(percent))
    print("__________________________________________________________________")

    # 01
    hopf_network_01 = CreateHopfieldNetwork(my_data01, num_of_examples01)
    data_after_changes01 = randomize_data(my_data01)
    result = check_me(hopf_network_01, my_data01, data_after_changes01)
    percent01 = check_if_succses(my_data01, result)
    print("The network has " + str(percent01) + "% success to remember model 01. It's a " + percent_check(percent01))
    #print(result)
    print("__________________________________________________________________")

    # 012
    hopf_network_012 = CreateHopfieldNetwork(my_data012, num_of_examples012)
    data_after_changes012 = randomize_data(my_data012)
    result = check_me(hopf_network_012, my_data012, data_after_changes012)
    percent012 = check_if_succses(my_data012, result)
    print(
        "The network has " + str(percent012) + "% success to remember model 012. It's a " + percent_check(percent012))
    print("__________________________________________________________________")

    # 0123
    hopf_network_0123 = CreateHopfieldNetwork(my_data0123, num_of_examples0123)
    data_after_changes0123 = randomize_data(my_data0123)
    result = check_me(hopf_network_0123, my_data0123, data_after_changes0123)
    percent0123 = check_if_succses(my_data0123, result)
    print(
        "The network has " + str(percent0123) + "% success to remember model 0123. It's a " + percent_check(percent0123))
    print("__________________________________________________________________")

    # 01234
    hopf_network_01234 = CreateHopfieldNetwork(my_data01234, num_of_examples01234)
    data_after_changes01234 = randomize_data(my_data01234)
    result = check_me(hopf_network_01234, my_data01234, data_after_changes01234)
    percent01234 = check_if_succses(my_data01234, result)
    print(
        "The network has " + str(percent01234) + "% success to remember model 0123. It's a " + percent_check(
            percent01234))
    print("__________________________________________________________________")

    # 012345
    hopf_network_012345 = CreateHopfieldNetwork(my_data012345, num_of_examples012345)
    data_after_changes012345 = randomize_data(my_data012345)
    result = check_me(hopf_network_012345, my_data012345, data_after_changes012345)
    percent012345 = check_if_succses(my_data012345, result)
    print("The network has " + str(percent012345) + "% success to remember model 012345. It's a " + percent_check(
        percent012345))
    print("__________________________________________________________________")

    # 0123456
    hopf_network_0123456 = CreateHopfieldNetwork(my_data0123456, num_of_examples0123456)
    data_after_changes0123456 = randomize_data(my_data0123456)
    result = check_me(hopf_network_0123456, my_data0123456, data_after_changes0123456)
    percent0123456 = check_if_succses(my_data0123456, result)
    print("The network has " + str(percent0123456) + "% success to remember model 0123456. It's a " + percent_check(
        percent0123456))
    print("__________________________________________________________________")

    # 01234567
    hopf_network_01234567 = CreateHopfieldNetwork(my_data01234567, num_of_examples01234567)
    data_after_changes01234567 = randomize_data(my_data01234567)
    result = check_me(hopf_network_01234567, my_data01234567, data_after_changes01234567)
    percent01234567 = check_if_succses(my_data01234567, result)
    print("The network has " + str(percent01234567) + "% success to remember model 01234567. It's a " + percent_check(
        percent01234567))
    print("__________________________________________________________________")

    # 012345678
    hopf_network_012345678 = CreateHopfieldNetwork(my_data012345678, num_of_examples012345678)
    data_after_changes012345678 = randomize_data(my_data012345678)
    result = check_me(hopf_network_012345678, my_data012345678, data_after_changes012345678)
    percent012345678 = check_if_succses(my_data012345678, result)
    print("The network has " + str(percent012345678) + "% success to remember model 012345678. It's a " + percent_check(
        percent012345678))
    print("__________________________________________________________________")

    # 0123456789
    hopf_network_0123456789 = CreateHopfieldNetwork(my_data0123456789, num_of_examples0123456789)
    data_after_changes0123456789 = randomize_data(my_data0123456789)
    result = check_me(hopf_network_0123456789, my_data0123456789, data_after_changes0123456789)
    percent0123456789 = check_if_succses(my_data0123456789, result)
    print("The network has " + str(percent0123456789) + "% success to remember model 0123456789. It's a " + percent_check(
        percent0123456789))
    print("__________________________________________________________________")
    time.sleep(10)