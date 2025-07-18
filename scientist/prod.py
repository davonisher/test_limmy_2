import time
import pickle
import os
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc

def save_cookies(driver, filename="google_scholar_cookies.pkl"):
    """Save cookies to file"""
    cookies = driver.get_cookies()
    with open(filename, 'wb') as f:
        pickle.dump(cookies, f)
    print(f"Cookies saved to {filename}")

def load_cookies(driver, filename="google_scholar_cookies.pkl"):
    """Load cookies from file"""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception as e:
                print(f"Error adding cookie: {e}")
        print(f"Cookies loaded from {filename}")
        return True
    return False

def check_login_status(driver):
    """Check if user is logged in to Google Scholar"""
    try:
        print("Checking login status...")
        
        # Go to Google Scholar homepage
        driver.get("https://scholar.google.com")
        time.sleep(3)
        
        print(f"Current URL: {driver.current_url}")
        
        # Check if we're redirected to sign-in page
        if "accounts.google.com" in driver.current_url or "signin" in driver.current_url:
            print("✗ Redirected to sign-in page - not logged in")
            return False
        
        # Check for various login indicators
        login_indicators = [
            '[data-email]',  # User email indicator
            '.gs_btn_act',   # Action buttons
            '.gs_btn_srt',   # Sort button
            '.gs_btn_gr_sml', # Group button
            'a[href*="citations?user="]', # User profile link
        ]
        
        for indicator in login_indicators:
            try:
                element = driver.find_element(By.CSS_SELECTOR, indicator)
                print(f"✓ Found login indicator: {indicator}")
                return True
            except NoSuchElementException:
                continue
        
        # If we get here, try to look for the search box (should be present when logged in)
        try:
            search_box = driver.find_element(By.ID, "gs_hdr_tsi")
            print("✓ Found search box - likely logged in")
            return True
        except NoSuchElementException:
            pass
        
        print("✗ No login indicators found")
        return False
                
    except Exception as e:
        print(f"Error checking login status: {e}")
        return False

def manual_login(driver):
    """Handle manual login process"""
    print("=" * 60)
    print("MANUAL LOGIN REQUIRED")
    print("=" * 60)
    print("1. A browser window will open")
    print("2. Please log in to your Google account manually")
    print("3. Navigate to Google Scholar and ensure you're logged in")
    print("4. The script will wait 1 minute for you to complete login")
    print("=" * 60)
    
    # Go to Google Scholar
    driver.get("https://scholar.google.com")
    
    # Wait 1 minute for user to complete login
    print("Waiting 1 minute for you to complete login...")
    time.sleep(30)
    
    # Check login status after 1 minute
    if check_login_status(driver):
        print("✓ Successfully logged in!")
        save_cookies(driver)
        return True
    else:
        print("✗ Login not detected after 1 minute")
        print("You can continue manually if needed")
        return False

def scrape_scientists_from_tab(driver, tab_index, csv_writer, label, max_pages=100):
    """Scrape scientists from a single tab, clicking next button every 5 seconds"""
    print(f"\n--- Tab {tab_index + 1} ({label}) ---")
    
    page_count = 0
    total_scientists = 0
    
    while page_count < max_pages:
        try:
            # Wait for page to load
            time.sleep(3)
            
            # Scrape all scientist cards on current page
            scientist_cards = driver.find_elements(By.CLASS_NAME, "gsc_1usr")
            
            if len(scientist_cards) == 0:
                print(f"Tab {tab_index + 1} ({label}): No more scientists found on page {page_count + 1}")
                break
            
            print(f"Tab {tab_index + 1} ({label}): Found {len(scientist_cards)} scientists on page {page_count + 1}")
            
            # Extract data from each scientist card
            for card in scientist_cards:
                scientist_info = {}
                
                # Extract name and profile URL
                try:
                    name_elem = card.find_element(By.CLASS_NAME, "gs_ai_name")
                    name_link = name_elem.find_element(By.TAG_NAME, "a")
                    scientist_info['name'] = name_link.text
                    scientist_info['profile_url'] = name_link.get_attribute('href')
                except Exception:
                    scientist_info['name'] = "N/A"
                    scientist_info['profile_url'] = "N/A"
                
                # Extract affiliation
                try:
                    affiliation_elem = card.find_element(By.CLASS_NAME, "gs_ai_aff")
                    scientist_info['affiliation'] = affiliation_elem.text
                except Exception:
                    scientist_info['affiliation'] = "N/A"
                
                # Extract email
                try:
                    email_elem = card.find_element(By.CLASS_NAME, "gs_ai_eml")
                    scientist_info['email'] = email_elem.text
                except Exception:
                    scientist_info['email'] = "N/A"
                
                # Extract citations
                try:
                    citations_elem = card.find_element(By.CLASS_NAME, "gs_ai_cby")
                    scientist_info['citations'] = citations_elem.text
                except Exception:
                    scientist_info['citations'] = "N/A"
                
                # Extract interests
                try:
                    interests_elem = card.find_element(By.CLASS_NAME, "gs_ai_int")
                    interest_links = interests_elem.find_elements(By.CLASS_NAME, "gs_ai_one_int")
                    interests = []
                    for interest in interest_links:
                        interests.append(interest.text)
                    scientist_info['interests'] = '; '.join(interests)
                except Exception:
                    scientist_info['interests'] = ""
                
                # Extract profile picture URL
                try:
                    img_elem = card.find_element(By.TAG_NAME, "img")
                    scientist_info['profile_picture_url'] = img_elem.get_attribute('src')
                    scientist_info['profile_picture_alt'] = img_elem.get_attribute('alt')
                except Exception:
                    scientist_info['profile_picture_url'] = "N/A"
                    scientist_info['profile_picture_alt'] = "N/A"
                
                # Add metadata
                scientist_info['label'] = label
                scientist_info['tab_index'] = tab_index + 1
                scientist_info['page_number'] = page_count + 1
                scientist_info['scraped_at'] = datetime.now().isoformat()
                
                # Write to CSV
                csv_writer.writerow(scientist_info)
                total_scientists += 1
                
                print(f"  - {scientist_info['name']} ({scientist_info['affiliation']})")
            
            # Look for next button
            try:
                next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next']")
                if next_button.is_enabled():
                    print(f"Tab {tab_index + 1} ({label}): Clicking next button...")
                    next_button.click()
                    page_count += 1
                    time.sleep(5)  # Wait 5 seconds before next page
                else:
                    print(f"Tab {tab_index + 1} ({label}): Next button disabled, reached end")
                    break
            except NoSuchElementException:
                print(f"Tab {tab_index + 1} ({label}): No next button found, reached end")
                break
            except Exception as e:
                print(f"Tab {tab_index + 1} ({label}): Error clicking next button: {e}")
                break
                
        except Exception as e:
            print(f"Tab {tab_index + 1} ({label}): Error on page {page_count + 1}: {e}")
            break
    
    print(f"Tab {tab_index + 1} ({label}): Completed! Scraped {total_scientists} scientists from {page_count + 1} pages")
    return total_scientists

def test_navigation(driver):
    """Test navigation to the AI scientists page and then to the AI label page, scraping 10 authors"""
    print("\n" + "=" * 60)
    print("TESTING NAVIGATION TO AI SCIENTISTS PAGE")
    print("=" * 60)
    
    try:
        # Step 1: Go to the basic search authors page
        base_url = "https://scholar.google.com/citations?view_op=search_authors"
        print(f"Navigating to: {base_url}")
        driver.get(base_url)
        time.sleep(60)
        print(f"Current URL after navigation: {driver.current_url}")

        # Check if we're on the correct page
        if "scholar.google.com/citations?view_op=search_authors" in driver.current_url:
            print("✓ Successfully on the search authors page")
            
            # Step 2: Go to the AI label authors page
            ai_url = "https://scholar.google.com/citations?view_op=search_authors&hl=en&mauthors=label:artificial_intelligence"
            print(f"Navigating to AI label page: {ai_url}")
            driver.get(ai_url)
            time.sleep(5)
            print(f"Current URL after navigation: {driver.current_url}")

            # Check if we're on the AI label page
            if "label:artificial_intelligence" in driver.current_url:
                print("✓ Successfully on the AI label page")
                
                # Scrape all author cards
                try:
                    scientist_cards = driver.find_elements(By.CLASS_NAME, "gsc_1usr")
                    print(f"✓ Found {len(scientist_cards)} scientist cards on AI label page")
                    if len(scientist_cards) == 0:
                        print("✗ No scientist cards found on AI label page")
                        return False

                    print(f"Scraping all {len(scientist_cards)} authors:")
                    scraped_data = []
                    
                    for i, card in enumerate(scientist_cards):
                        scientist_info = {}
                        
                        # Extract name
                        try:
                            name_elem = card.find_element(By.CLASS_NAME, "gs_ai_name")
                            name_link = name_elem.find_element(By.TAG_NAME, "a")
                            scientist_info['name'] = name_link.text
                            scientist_info['profile_url'] = name_link.get_attribute('href')
                        except Exception:
                            scientist_info['name'] = "N/A"
                            scientist_info['profile_url'] = "N/A"
                        
                        # Extract affiliation
                        try:
                            affiliation_elem = card.find_element(By.CLASS_NAME, "gs_ai_aff")
                            scientist_info['affiliation'] = affiliation_elem.text
                        except Exception:
                            scientist_info['affiliation'] = "N/A"
                        
                        # Extract email (if available)
                        try:
                            email_elem = card.find_element(By.CLASS_NAME, "gs_ai_eml")
                            scientist_info['email'] = email_elem.text
                        except Exception:
                            scientist_info['email'] = "N/A"
                        
                        # Extract citations
                        try:
                            citations_elem = card.find_element(By.CLASS_NAME, "gs_ai_cby")
                            scientist_info['citations'] = citations_elem.text
                        except Exception:
                            scientist_info['citations'] = "N/A"
                        
                        # Extract interests/tags
                        try:
                            interests_elem = card.find_element(By.CLASS_NAME, "gs_ai_int")
                            interest_links = interests_elem.find_elements(By.CLASS_NAME, "gs_ai_one_int")
                            interests = []
                            for interest in interest_links:
                                interests.append(interest.text)
                            scientist_info['interests'] = interests
                        except Exception:
                            scientist_info['interests'] = []
                        
                        # Extract profile picture URL
                        try:
                            img_elem = card.find_element(By.TAG_NAME, "img")
                            scientist_info['profile_picture_url'] = img_elem.get_attribute('src')
                            scientist_info['profile_picture_alt'] = img_elem.get_attribute('alt')
                        except Exception:
                            scientist_info['profile_picture_url'] = "N/A"
                            scientist_info['profile_picture_alt'] = "N/A"
                        
                        # Add to scraped data
                        scraped_data.append(scientist_info)
                        
                        # Print progress
                        print(f"{i+1}. {scientist_info['name']} - {scientist_info['affiliation']} - {scientist_info['citations']}")
                        print(f"   Interests: {', '.join(scientist_info['interests'])}")
                        print(f"   Profile: {scientist_info['profile_url']}")
                        print(f"   Picture: {scientist_info['profile_picture_url']}")
                        print()
                    
                    print(f"✓ Successfully scraped {len(scraped_data)} AI scientists!")
                    print(f"Total data points collected: {len(scraped_data)}")
                    return True

                except Exception as e:
                    print(f"Error looking for scientist cards: {e}")
                    return False
            else:
                print("✗ Not on the AI label page")
                return False
        else:
            print("✗ Not on the search authors page")
            return False

    except Exception as e:
        print(f"Error during navigation test: {e}")
        return False

def main():
    print("Starting Google Scholar Login Test")
    print("=" * 60)
    
    try:
        # Initialize driver
        print("Initializing undetected Chrome driver...")
        
        options = uc.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        driver = uc.Chrome(options=options)
        driver.set_window_size(1920, 1080)
        
        print("✓ Chrome driver initialized successfully")
        
        # Try to load existing cookies
        cookies_loaded = load_cookies(driver)
        
        # Check if we're logged in
        if not cookies_loaded or not check_login_status(driver):
            print("\nNo valid session found. Starting manual login process...")
            if not manual_login(driver):
                print("Failed to log in. Exiting.")
                return
        else:
            print("✓ Using existing session")
        
        # Test navigation to AI scientists page and scrape authors
        if test_navigation(driver):
            print("\n✓ All tests passed! Ready for scraping.")
            
            # Start concurrent scraping with 5 tabs
            print("\n" + "=" * 60)
            print("STARTING CONCURRENT SCRAPING WITH 5 TABS")
            print("=" * 60)
            
            # Create CSV file for saving data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"ai_scientists_multilabel_{timestamp}.csv"
            
            # Define fieldnames for CSV
            fieldnames = [
                'name', 'profile_url', 'affiliation', 'email', 'citations', 
                'interests', 'profile_picture_url', 'profile_picture_alt',
                'label', 'tab_index', 'page_number', 'scraped_at'
            ]
            
            # Open CSV file and start scraping
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
                
                # Define 5 different labels to scrape
                labels = [
                    #"artificial_intelligence",
                    #"computer_vision", 
                    "robotics",
                    "machine_learning",
                    "autonomous_driving",
                    "deep_learning",
              
                ]
                
                num_labels = len(labels)
                print(f"Using {num_labels} labels: {labels}")
                
                # Open additional tabs with different labels (we already have 1)
                for i in range(num_labels - 1):  # -1 because we already have the first tab
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[-1])
                    label_url = f"https://scholar.google.com/citations?view_op=search_authors&hl=en&mauthors=label:{labels[i+1]}"
                    driver.get(label_url)
                    time.sleep(2)
                    print(f"✓ Tab {i+2} opened with {labels[i+1]} label page")
                
                # Switch back to first tab and load artificial_intelligence
                driver.switch_to.window(driver.window_handles[0])
                ai_url = f"https://scholar.google.com/citations?view_op=search_authors&hl=en&mauthors=label:{labels[0]}"
                driver.get(ai_url)
                time.sleep(2)
                print(f"✓ Tab 1 loaded with {labels[0]} label page")
                print(f"✓ All {num_labels} tabs ready for scraping")
                
                # Scrape from each tab concurrently
                print(f"\nStarting concurrent scraping across all {num_labels} tabs...")
                total_scientists = 0
                page_count = 0
                max_pages = 100
                
                while page_count < max_pages:
                    print(f"\n--- Processing page {page_count + 1} across all tabs ---")
                    
                    # Process all tabs for current page
                    for tab_index in range(num_labels):
                        try:
                            driver.switch_to.window(driver.window_handles[tab_index])
                            time.sleep(1)  # Brief wait for tab switch
                            
                            # Scrape scientists from current page
                            scientist_cards = driver.find_elements(By.CLASS_NAME, "gsc_1usr")
                            
                            if len(scientist_cards) > 0:
                                print(f"Tab {tab_index + 1} ({labels[tab_index]}): Found {len(scientist_cards)} scientists")
                                
                                # Extract data from each scientist card
                                for card in scientist_cards:
                                    scientist_info = {}
                                    
                                    # Extract name and profile URL
                                    try:
                                        name_elem = card.find_element(By.CLASS_NAME, "gs_ai_name")
                                        name_link = name_elem.find_element(By.TAG_NAME, "a")
                                        scientist_info['name'] = name_link.text
                                        scientist_info['profile_url'] = name_link.get_attribute('href')
                                    except Exception:
                                        scientist_info['name'] = "N/A"
                                        scientist_info['profile_url'] = "N/A"
                                    
                                    # Extract affiliation
                                    try:
                                        affiliation_elem = card.find_element(By.CLASS_NAME, "gs_ai_aff")
                                        scientist_info['affiliation'] = affiliation_elem.text
                                    except Exception:
                                        scientist_info['affiliation'] = "N/A"
                                    
                                    # Extract email
                                    try:
                                        email_elem = card.find_element(By.CLASS_NAME, "gs_ai_eml")
                                        scientist_info['email'] = email_elem.text
                                    except Exception:
                                        scientist_info['email'] = "N/A"
                                    
                                    # Extract citations
                                    try:
                                        citations_elem = card.find_element(By.CLASS_NAME, "gs_ai_cby")
                                        scientist_info['citations'] = citations_elem.text
                                    except Exception:
                                        scientist_info['citations'] = "N/A"
                                    
                                    # Extract interests
                                    try:
                                        interests_elem = card.find_element(By.CLASS_NAME, "gs_ai_int")
                                        interest_links = interests_elem.find_elements(By.CLASS_NAME, "gs_ai_one_int")
                                        interests = []
                                        for interest in interest_links:
                                            interests.append(interest.text)
                                        scientist_info['interests'] = '; '.join(interests)
                                    except Exception:
                                        scientist_info['interests'] = ""
                                    
                                    # Extract profile picture URL
                                    try:
                                        img_elem = card.find_element(By.TAG_NAME, "img")
                                        scientist_info['profile_picture_url'] = img_elem.get_attribute('src')
                                        scientist_info['profile_picture_alt'] = img_elem.get_attribute('alt')
                                    except Exception:
                                        scientist_info['profile_picture_url'] = "N/A"
                                        scientist_info['profile_picture_alt'] = "N/A"
                                    
                                    # Add metadata
                                    scientist_info['label'] = labels[tab_index]
                                    scientist_info['tab_index'] = tab_index + 1
                                    scientist_info['page_number'] = page_count + 1
                                    scientist_info['scraped_at'] = datetime.now().isoformat()
                                    
                                    # Write to CSV
                                    csv_writer.writerow(scientist_info)
                                    total_scientists += 1
                                    
                                    print(f"  - {scientist_info['name']} ({scientist_info['affiliation']})")
                            else:
                                print(f"Tab {tab_index + 1} ({labels[tab_index]}): No scientists found")
                                
                        except Exception as e:
                            print(f"Error processing tab {tab_index + 1}: {e}")
                    
                    # Click Next button on all tabs simultaneously
                    print(f"\nClicking Next button on all tabs (page {page_count + 1})...")
                    next_buttons_clicked = 0
                    
                    for tab_index in range(num_labels):
                        try:
                            driver.switch_to.window(driver.window_handles[tab_index])
                            time.sleep(0.5)  # Brief wait for tab switch
                            
                            next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next']")
                            if next_button.is_enabled():
                                next_button.click()
                                next_buttons_clicked += 1
                                print(f"  ✓ Tab {tab_index + 1} ({labels[tab_index]}): Next clicked")
                            else:
                                print(f"  ✗ Tab {tab_index + 1} ({labels[tab_index]}): Next button disabled")
                        except NoSuchElementException:
                            print(f"  ✗ Tab {tab_index + 1} ({labels[tab_index]}): No next button found")
                        except Exception as e:
                            print(f"  ✗ Tab {tab_index + 1} ({labels[tab_index]}): Error clicking next: {e}")
                    
                    # If no next buttons were clicked, we've reached the end
                    if next_buttons_clicked == 0:
                        print("\nAll tabs have reached the end. Stopping scraping.")
                        break
                    
                    page_count += 1
                    print(f"\nWaiting 2 seconds before next page...")
                    time.sleep(2)  # Wait 2 seconds before next iteration
                
                print(f"\n" + "=" * 60)
                print(f"SCRAPING COMPLETED!")
                print(f"Total scientists scraped: {total_scientists}")
                print(f"Data saved to: {csv_filename}")
                print("=" * 60)
            
        else:
            print("\n✗ Navigation test failed.")
        
        # Keep browser open for manual inspection
        print("\nBrowser will remain open for manual inspection...")
        print("Press Ctrl+C to close the browser when you're done.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClosing browser...")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        try:
            driver.quit()
            print("Browser closed.")
        except:
            pass

if __name__ == "__main__":
    main() 