#!/bin/bash
# g++ -c headers/activation_functions.cpp
# g++ -c headers/preprocessing.cpp
# g++ -c main.cpp
g++ -c *.cpp headers/*.cpp
g++ -o main activation_functions.o preprocessing.o main.o
# g++ -o main *.o

mkdir build
mv main build/
mv *.o build/  
cd build  
# Run the compiled program
./main

# Clean up object files
rm *.o
# Clean up the executable
rm main
# Clean up the build directory
cd ..
# rm -rf build
# End of file: DeepLearningLibrary/compiler_script.sh


# echo ""
# echo "Compilation and execution completed successfully."
# echo "Build directory and object files have been cleaned up."
# echo "Thank you for using the Deep Learning CPP Library!"
# echo "For any issues, please refer to the README or contact the developer."
# echo "Have a great day!"    
# echo ""
# echo "You can also modify the source files and recompile as needed."
# echo "For further development, consider adding more features or optimizations."
# echo "Remember to keep your code organized and well-documented."
# echo "If you encounter any bugs or issues, please report them on the project's GitHub page."
# echo "Your contributions and feedback are greatly appreciated!"
