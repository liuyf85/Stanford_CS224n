# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

# 都说了要用现成的代码，那就去找什么函数符合我们的需求
# utils 一般都是一些处理杂事的函数的定义
import utils

# 找到了现有的函数，我们需要模拟一个 prediction 来调用这个函数
places = ["London"] * len(open("birth_dev.tsv").readlines() )
                          
total, correct = utils.evaluate_places(filepath = "birth_dev.tsv", predicted_places = places)
# python 要输出百分号需要 %% ，因为一个被用作格式化输出
print("%.2f%%"%(correct / total * 100))