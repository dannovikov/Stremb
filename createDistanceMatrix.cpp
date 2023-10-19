#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <N>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile) {
        std::cerr << "Unable to open input file\n";
        return 1;
    }

    int N = std::stoi(argv[3]);
    std::unordered_map<std::string, int> sequenceMap;
    std::vector<std::vector<float>> distances(N, std::vector<float>(N, 0.0f));
    std::string line;
    std::getline(infile, line); // Read and discard the header line

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string source, target, dist_str;
        std::getline(iss, source, ',');
        std::getline(iss, target, ',');
        std::getline(iss, dist_str);

        if (sequenceMap.find(source) == sequenceMap.end()) {
            sequenceMap[source] = sequenceMap.size();
        }

        if (sequenceMap.find(target) == sequenceMap.end()) {
            sequenceMap[target] = sequenceMap.size();
        }

        int srcIndex = sequenceMap[source];
        int tgtIndex = sequenceMap[target];
        float dist = std::stof(dist_str);

        distances[srcIndex][tgtIndex] = dist;
        distances[tgtIndex][srcIndex] = dist;
    }

    infile.close();

    std::ofstream outfile(argv[2]);
    if (!outfile) {
        std::cerr << "Unable to open output file\n";
        return 1;
    }

    for (const auto& row : distances) {
        for (size_t i = 0; i < row.size() - 1; ++i) {
            outfile << row[i] << ",";
        }
        outfile << row.back() << "\n";
    }

    outfile.close();
    std::cout << "Distance matrix written to " << argv[2] << "\n";
    return 0;
}
