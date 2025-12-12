
import os
import shutil
import sys

def organize_reports():
    # Define source and destination directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(base_dir, 'build')
    shortened_build_dir = os.path.join(base_dir, 'shortned_build')

    # Create destination directory if it doesn't exist
    if not os.path.exists(shortened_build_dir):
        os.makedirs(shortened_build_dir)
        print(f"Created directory: {shortened_build_dir}")
    else:
        print(f"Directory already exists: {shortened_build_dir}")

    # Check if build directory exists
    if not os.path.exists(build_dir):
        print(f"Error: Build directory not found at {build_dir}")
        return

    # Iterate through items in the build directory
    items = os.listdir(build_dir)
    count = 0

    for item in items:
        item_path = os.path.join(build_dir, item)

        # We are only interested in directories
        if os.path.isdir(item_path):
            # Construct the path to the report directory
            # Structure: <project_name>/out.prj/solution1/syn/report/
            report_dir = os.path.join(item_path, 'out.prj', 'solution1', 'syn', 'report')
            
            # Check for kernel.cpp in the project root
            kernel_src = os.path.join(item_path, 'kernel.cpp')

            # We process if either report dict exists OR kernel.cpp exists (though usually they go together)
            has_reports = os.path.exists(report_dir) and os.path.isdir(report_dir)
            has_kernel = os.path.exists(kernel_src)

            if has_reports or has_kernel:
                print(f"Processing {item}...")
                
                # Create a subfolder in shortened_build with the project name (item)
                dest_dir = os.path.join(shortened_build_dir, item)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                copied_files = 0
                
                # Copy report files
                if has_reports:
                    report_files = os.listdir(report_dir)
                    for filename in report_files:
                        src_file = os.path.join(report_dir, filename)
                        dest_file = os.path.join(dest_dir, filename)
                        
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dest_file)
                            copied_files += 1
                
                # Copy kernel.cpp
                if has_kernel:
                    shutil.copy2(kernel_src, os.path.join(dest_dir, 'kernel.cpp'))
                    print(f"  Copied kernel.cpp")
                    copied_files += 1
                
                print(f"  Total copied files: {copied_files}")
                count += 1

    print(f"\nFinished organizing reports. Processed {count} project(s).")

if __name__ == "__main__":
    organize_reports()
