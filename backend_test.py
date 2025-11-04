#!/usr/bin/env python3
"""
SkyAR Backend API Test Script
Tests the SkyAR demo application API endpoints
"""

import requests
import os
import time
import sys
from pathlib import Path

class SkyARTester:
    def __init__(self, base_url="http://localhost:8002"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.video_id = None
        
    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, json_data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                elif json_data:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=json_data, headers=headers)
                else:
                    response = requests.post(url, data=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"Response: {response.text}")
                    return False, response.json()
                except:
                    return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}
    
    def test_health_check(self):
        """Test the health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "health",
            200
        )
        
        if success:
            expected = {"status": "healthy", "service": "SkyAR Demo"}
            if response == expected:
                print("✅ Health check response matches expected format")
                return True
            else:
                print(f"❌ Health check response doesn't match expected format: {response}")
                return False
        return False
    
    def test_get_templates(self):
        """Test getting available sky templates"""
        success, response = self.run_test(
            "Get Sky Templates",
            "GET",
            "api/templates",
            200
        )
        
        if success and "templates" in response:
            templates = response["templates"]
            if len(templates) == 6:
                print(f"✅ Found 6 sky templates: {', '.join(templates.keys())}")
                return True
            else:
                print(f"❌ Expected 6 templates, got {len(templates)}")
                return False
        return False
    
    def test_upload_video(self, video_path):
        """Test uploading a video file"""
        if not Path(video_path).exists():
            print(f"❌ Test video not found at {video_path}")
            return False
        
        with open(video_path, 'rb') as video_file:
            files = {'file': (Path(video_path).name, video_file, 'video/mp4')}
            success, response = self.run_test(
                "Upload Video",
                "POST",
                "api/upload",
                200,
                files=files
            )
        
        if success and "video_id" in response:
            self.video_id = response["video_id"]
            print(f"✅ Video uploaded successfully with ID: {self.video_id}")
            return True
        return False
    
    def test_process_video(self, sky_template="floatingcastle"):
        """Test starting video processing"""
        if not self.video_id:
            print("❌ No video ID available for processing")
            return False
        
        json_data = {
            "video_id": self.video_id,
            "sky_template": sky_template,
            "auto_light_matching": False,
            "relighting_factor": 0.8,
            "recoloring_factor": 0.5,
            "halo_effect": True
        }
        
        success, response = self.run_test(
            "Process Video",
            "POST",
            "api/process",
            200,
            json_data=json_data
        )
        
        if success and response.get("success") == True:
            print(f"✅ Video processing started successfully")
            return True
        return False
    
    def test_status_check(self, max_attempts=30):
        """Test checking processing status"""
        if not self.video_id:
            print("❌ No video ID available for status check")
            return False
        
        print(f"\n🔍 Monitoring processing status (max {max_attempts} attempts)...")
        
        for attempt in range(max_attempts):
            success, response = self.run_test(
                f"Status Check (Attempt {attempt+1}/{max_attempts})",
                "GET",
                f"api/status/{self.video_id}",
                200
            )
            
            if success:
                status = response.get("status")
                progress = response.get("progress", 0)
                message = response.get("message", "")
                
                print(f"Status: {status}, Progress: {progress}%, Message: {message}")
                
                if status == "completed":
                    print("✅ Processing completed successfully!")
                    return True
                elif status == "error":
                    print(f"❌ Processing failed: {message}")
                    return False
                
                # Wait before next check
                time.sleep(2)
            else:
                return False
        
        print("❌ Processing did not complete within the expected time")
        return False
    
    def test_download_result(self):
        """Test downloading the processed video"""
        if not self.video_id:
            print("❌ No video ID available for download")
            return False
        
        # This is a bit different as we're downloading a file
        url = f"{self.base_url}/api/download/{self.video_id}"
        print(f"\n🔍 Testing Download Result...")
        
        try:
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                # Save the file to verify it's a valid video
                output_path = f"/tmp/test_download_{self.video_id}.mp4"
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Check if file exists and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"✅ Downloaded video successfully to {output_path}")
                    self.tests_passed += 1
                    return True
                else:
                    print("❌ Downloaded file is empty or missing")
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                print(f"Response: {response.text}")
        
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
        
        self.tests_run += 1
        return False

def main():
    # Setup
    base_url = "http://localhost:8002"  # Default URL
    tester = SkyARTester(base_url)
    test_video = "/app/test_videos/annarbor.mp4"  # Path to test video
    
    # Run tests
    if not tester.test_health_check():
        print("❌ Health check failed, stopping tests")
        return 1
    
    if not tester.test_get_templates():
        print("❌ Failed to get sky templates, stopping tests")
        return 1
    
    if not tester.test_upload_video(test_video):
        print("❌ Video upload failed, stopping tests")
        return 1
    
    if not tester.test_process_video():
        print("❌ Video processing failed to start, stopping tests")
        return 1
    
    if not tester.test_status_check():
        print("❌ Video processing status check failed")
        return 1
    
    if not tester.test_download_result():
        print("❌ Video download failed")
        return 1
    
    # Print results
    print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    sys.exit(main())
