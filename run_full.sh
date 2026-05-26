pip install -e .

cd ./docs
make clean

cd ../
doxygen
cd ./docs

mkdir -p ./build/html/_static/css
cat <<EOF > ./build/html/_static/css/customWidth.css
.wy-nav-content {
    max-width: 1400px !important;
}
EOF

make html