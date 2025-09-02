import csv
import os
import logging
from uncomp.utils.logger import Logger

logger = Logger()

def save_lists_to_csv(dictionary, keys_to_save, output_directory):
    # 确保输出目录存在
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        logger.info(f"mkdir -p {output_directory}")
  
    for key in keys_to_save:
        # logger.debug(f"key is: {key}")
        # logger.debug(f"dictionary[key] is: {dictionary[key]}")
        
        if key in dictionary and isinstance(dictionary[key], list) and len(dictionary[key]) == 32:
            output_file = os.path.join(output_directory, f"{key}.csv")
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 写入32*32的列表
                for row in dictionary[key]:
                    if len(row) == 32:
                        writer.writerow(row)
                    else:
                        print(f"警告: {key} 的行 {row} 不是32个元素")
            print(f"CSV文件已保存为 {output_file}")
        else:
            print(f"警告: {key} 不存在或不是32*32的列表")