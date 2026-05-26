cd ./docs
rm -rf ./build/html

mkdir -p ./build/html/_static/css
cat <<EOF > ./build/html/_static/css/customWidth.css
.wy-nav-content {
    max-width: 1400px !important;
}
EOF

make html