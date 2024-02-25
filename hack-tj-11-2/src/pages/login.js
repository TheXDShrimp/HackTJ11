// pages/login.js

'use client';

import { useRouter } from 'next/router';
import Navbar from '../components/Navbar';
import Link from 'next/link';
import BIRDS from 'vanta/dist/vanta.birds.min';
import React, { useEffect, useRef, useState } from 'react'

function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const [vantaEffect, setVantaEffect] = useState(null)
  const myRef = useRef(null)
  useEffect(() => {
    if (!vantaEffect && myRef && myRef.current) {
      setVantaEffect(BIRDS({
        el: myRef.current
      }))
    }
    return () => {
      if (vantaEffect) vantaEffect.destroy()
    }
  }, [vantaEffect])

  // // Function to handle login with Google OAuth
  // const handleLoginWithGoogle = () => {
  //   // Redirect the user to Google OAuth login page
  //   window.location.href = '/api/auth/google'; // Handle this route in your API routes
  // };

  // // Function to handle username generation from name
  // const generateUsername = (name) => {
  //   // Logic to convert name to username (e.g., lowercase and remove spaces)
  //   return name.toLowerCase().replace(/\s+/g, '_');
  // };

  // // Function to generate a random password
  // const generatePassword = () => {
  //   // Logic to generate a random password (e.g., using Math.random())
  //   return Math.random().toString(36).slice(-8);
  // };

  // Function to handle form submission (not shown here)
  const handleSubmit = (event) => {
    event.preventDefault();

    router.push('/dashboard')
    // try {
    //   const response = await fetch('http://127.0.0.1:5000/login', {
    //     method: 'POST',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({
    //       username,
    //       password,
    //     }),
    //   });

    //   if (response.ok) {
    //     // Handle successful login response
    //     const data = await response.json();
    //     // Redirect or perform any necessary actions after successful login
    //     console.log('Login successful:', data);
    //   } else {
    //     // Handle unsuccessful login response
    //     const errorData = await response.json();
    //     console.error('Login failed:', errorData);
    //   }
    // } catch (error) {
    //   console.error('Error during login:', error);
    // }
  };


  return (
    <>
      <Navbar />
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/vanta/dist/vanta.waves.min.js"></script>
      <div ref={myRef} className="z-10 min-h-screen justify-center items-center p-36 my-auto text-black">
        <div className="p-12 bg-gray-100 rounded-lg shadow-md mx-80">
          <h1 className="font-bold text-5xl text-center mb-12">Log In</h1>
          {/* <button className="text-center mx-auto" onClick={handleLoginWithGoogle}>Login with Google</button> */}
          <form className="max-w-sm mx-auto" onSubmit={handleSubmit}>
            <div className="mb-5">
              <label htmlFor="email" className="block mb-2 text-md font-medium text-gray-900">Email</label>
              <input type="email" value={username} onChange={(e) => setUsername(e.target.value)} id="email" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5" placeholder="name@aegis.com" required />
            </div>
            <div className="mb-5">
              <label htmlFor="password" className="block mb-2 text-md font-medium text-gray-900">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} id="password" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5" placeholder="•••••••••" required />
            </div>
            <div className="flex items-start mb-5">
              {/* <div className="flex items-center h-5">
                <input id="remember" type="checkbox" value="" className="w-4 h-4 border border-gray-300 rounded bg-gray-50 focus:ring-3 focus:ring-blue-300 dark:bg-gray-700 dark:border-gray-600 dark:focus:ring-blue-600 dark:ring-offset-gray-800 dark:focus:ring-offset-gray-800" required />
              </div> */}
              <p className="text-sm font-medium text-gray-900">Don't have an account? <Link href="/signup" className="font-bold text-blue-500">Create an account.</Link></p>
            </div>
            <button type="submit" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm w-full px-5 py-2.5 text-center">Submit</button>
          </form>
        </div>
      </div>
    </>
  );
}

export default LoginPage;
