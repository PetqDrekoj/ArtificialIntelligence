class Repository:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.load_data()

    def load_data(self):
        fileDescriptor = open(self.input_file, "r")
        self.d = []
        for line in fileDescriptor:
            if(line.find('#') == -1):
                attributes = line.split(",")
                self.d.append({'Texture': float(attributes[0]),
                          'Capacity': float(attributes[1])})

    def get_input_data(self):
        return self.d

    def store_data(self, results):
        fileDescriptor = open(self.output_file, "w")
        for r in results:
            fileDescriptor.write(str(r))
            fileDescriptor.write("\n")
