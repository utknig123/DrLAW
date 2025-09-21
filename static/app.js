// static/app.js
const BACKEND_URL = 'https://drlaw-backend.onrender.com';

// Handle client-side routing
function handleRouting() {
    const path = window.location.pathname;
    
    // Check if user is logged in (simple check)
    const isLoggedIn = localStorage.getItem('userToken') || sessionStorage.getItem('userToken');
    
    if (path === '/' || path === '/index.html') {
        if (!isLoggedIn) {
            window.location.href = '/front.html';
        }
    } else if (path === '/front.html' || path === '/login.html') {
        if (isLoggedIn) {
            window.location.href = '/index.html';
        }
    }
}

// Handle form submissions
document.addEventListener('DOMContentLoaded', function() {
    handleRouting();
    
    // Signup form
    const signupForm = document.querySelector('form[action*="signup"]');
    if (signupForm) {
        signupForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            try {
                const response = await fetch(`${BACKEND_URL}/signup`, {
                    method: 'POST',
                    body: formData,
                    credentials: 'include'
                });
                
                if (response.ok) {
                    window.location.href = '/index.html';
                } else {
                    alert('Signup failed');
                }
            } catch (error) {
                console.error('Signup error:', error);
            }
        });
    }

    // Login form
    const loginForm = document.querySelector('form[action*="login"]');
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            try {
                const response = await fetch(`${BACKEND_URL}/login`, {
                    method: 'POST',
                    body: formData,
                    credentials: 'include'
                });
                
                if (response.ok) {
                    window.location.href = '/index.html';
                } else {
                    alert('Login failed');
                }
            } catch (error) {
                console.error('Login error:', error);
            }
        });
    }

    // Logout functionality
    const logoutLinks = document.querySelectorAll('a[href*="logout"]');
    logoutLinks.forEach(link => {
        link.addEventListener('click', async function(e) {
            e.preventDefault();
            try {
                await fetch(`${BACKEND_URL}/logout`, {
                    method: 'POST',
                    credentials: 'include'
                });
                localStorage.removeItem('userToken');
                sessionStorage.removeItem('userToken');
                window.location.href = '/front.html';
            } catch (error) {
                console.error('Logout error:', error);
            }
        });
    });

    // Chat functionality
    const askForm = document.getElementById('ask-form');
    if (askForm) {
        askForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const questionInput = document.getElementById('question-input');
            const question = questionInput.value.trim();
            
            if (!question) return;
            
            try {
                const response = await fetch(`${BACKEND_URL}/ask`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        question: question,
                        language: 'en'
                    }),
                    credentials: 'include'
                });
                
                const data = await response.json();
                console.log('Response:', data);
            } catch (error) {
                console.error('Chat error:', error);
            }
        });
    }
});