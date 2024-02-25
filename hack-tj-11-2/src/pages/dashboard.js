// pages/login.js

'use client';

import { useState } from 'react';
import { useRouter } from 'next/router';
import Navbar from '../components/Navbar';
import Link from 'next/link';
import Image from 'next/image';
import { Slider } from "@material-tailwind/react";

function Dashboard() {
  const router = useRouter();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [num, setNum] = useState(50);
  const [strength, setStrength] = useState(12);

  const onSubmit = () => {
    console.log('submit')
  }

  return (
    <>
      <Navbar loggedIn={true} />
      <div className="min-h-screen p-24 py-36 py-auto bg-black text-white">
        <h1 className="font-bold text-5xl mb-12">Welcome, <span className="text-blue-200">Sritan Motati</span>.</h1>
        <div className="grid grid-cols-4 gap-6">
          <div className="col-span-1 justify-center rounded-full">
            <h2 className="text-4xl font-bold mb-8">Settings</h2>
            <Image
              src="/sritan-motati-pfp.jpg"
              className="rounded-full"
              width={256}
              height={256}
            />
          </div>
          <div className="col-span-3">
            <form onSubmit={onSubmit}>
                <div className="mb-6">
                    <label htmlFor="password" className="block mb-2 text-md font-medium text-gray-900 text-white">Password</label>
                    <input type="password" id="password" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" placeholder="•••••••••" value={password} onChange={(e) => setPassword(e.target.value)} required />
                </div> 
                <div className="mb-6">
                    <label htmlFor="confirm_password" className="block mb-2 text-md font-medium text-gray-900 text-white">Confirm password</label>
                    <input type="password" id="confirm_password" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" placeholder="•••••••••" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required />
                </div>
                <div className="grid gap-6 mb-6 md:grid-cols-2">
                    <div>
                        <label htmlFor="num_iterations" className="block mb-2 text-md font-medium text-gray-900 text-white">Number of iterations (1-100)</label>
                        <input type="number" id="num_iterations" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" value={num} onChange={(e) => setNum(e.target.value)} min={1} max={100} required />
                        {/* <Slider className="w-full" min={1} max={100} defaultValue={50} barClassName='bg-blue-500' /> */}
                    </div>
                    <div>
                        <label htmlFor="strength" className="block mb-2 text-md font-medium text-gray-900 text-white">Strength of attack (1-100)</label>
                        <input type="number" id="strength" className="bg-gray-50 border border-gray-300 text-gray-900 text-md rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500" value={strength} onChange={(e) => setStrength(e.target.value)} min={1} max={100} required />
                    </div>
                </div>
                <button type="submit" className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full sm:max-w-48 px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Submit</button>
            </form>
          </div>
        </div>
        <h2 className="text-4xl font-bold mt-20 mb-8">Previous Runs</h2>
        <div className="grid gap-6 grid-cols-4">
          <div className="p-6 rounded-lg shadow-md bg-slate-800">
            <p className="text-md text-gray-400">December 30, 1984</p>
            <h2 className='text-xl mt-2 mb-6 font-semibold'>[file name]</h2>
            <button className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Download</button>
          </div>
          <div className="p-6 rounded-lg shadow-md bg-slate-800">
            <p className="text-md text-gray-400">December 30, 1984</p>
            <h2 className='text-xl mt-2 mb-6 font-semibold'>[file name]</h2>
            <button className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Download</button>
          </div>
          <div className="p-6 rounded-lg shadow-md bg-slate-800">
            <p className="text-md text-gray-400">December 30, 1984</p>
            <h2 className='text-xl mt-2 mb-6 font-semibold'>[file name]</h2>
            <button className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Download</button>
          </div>
          <div className="p-6 rounded-lg shadow-md bg-slate-800">
            <p className="text-md text-gray-400">December 30, 1984</p>
            <h2 className='text-xl mt-2 mb-6 font-semibold'>[file name]</h2>
            <button className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-md w-full px-5 py-2.5 text-center bg-blue-600 hover:bg-blue-700 focus:ring-blue-800">Download</button>
          </div>
        </div>
      </div>
    </>
  );
}

export default Dashboard;
