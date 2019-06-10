
import os
import sys

class IO:
    """
    Input/output class mainly for writing data to disk. You should write to the output directory
    but this can handle writing/reading anywhere.

    """
    def __init__(self):
        self.files = {}

    def reInit(self):
        """
        Reinitialize the files, so we can append rather than re-write them.
        """
        for key, val in self.files.items():
            self.files[key] = open(key, 'a')

    def writer(self, filename, data, header=False):
        """
        Write to a file called filename. If not created, this class will open it for you.

        Parameters
        ----------
        filename : Name of the file you would like to write to.
        data : A tuple of data that will be written to the file.
        header : If you would like to write a string to the file, use the header. This allows one to comment the data files. Default: False.
        """
        if filename not in self.files.keys():
            self.files[filename] = open(filename, 'w')
    
        if header:
            self.files[filename].write(data)
        else:
            for elem in data:
                self.files[filename].write('%1.20e\t' % (elem))
        self.files[filename].write('\n')
        
        self.files[filename].flush()

    def close(self, filename):
        self.files[filename].close()

    def __del__(self):
        for key in self.files:
            self.files[key].close()
