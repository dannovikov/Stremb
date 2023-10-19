# Save this code in a file named pread.py

from multiprocessing import Process, Manager, Value, Queue
from subprocess import check_output
import os

def get_line_count(file_path):
    return int(check_output(['wc', '-l', file_path]).split()[0])

def process_chunk(start_line, end_line, file_path, queue):
    with open(file_path, 'r') as f:
        for _ in range(start_line):
            next(f)
        for line in (line for i, line in enumerate(f) if start_line <= i < end_line):
            queue.put(line)

class ParallelReader:
    def __init__(self, file_path, num_processes=4):
        self.file_path = file_path
        self.num_processes = num_processes
        self.queue = Queue()
        self.processes = []

    def start(self):
        total_lines = get_line_count(self.file_path)
        lines_per_chunk = total_lines // self.num_processes

        for i in range(self.num_processes):
            start_line = i * lines_per_chunk
            end_line = (i + 1) * lines_per_chunk if i < self.num_processes - 1 else total_lines
            p = Process(target=process_chunk, args=(start_line, end_line, self.file_path, self.queue))
            self.processes.append(p)
            p.start()

    def stop(self):
        for p in self.processes:
            p.join()

def parallel_read(file_path, num_processes=4):
    reader = ParallelReader(file_path, num_processes)
    reader.start()
    while True:
        line = reader.queue.get()
        if line is None:
            break
        yield line
    reader.stop()

# Usage example:
# from pread import parallel_read
# for line in parallel_read('your_file.txt'):
#     parse(line)
