for file in $(find -name "*.py" -type f); do
    awk -f "make_student_version.awk" "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
done
