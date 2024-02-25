'use client';

import Image from "next/image";
import React, { useState, useEffect, useRef } from 'react'
import Navbar from "../components/Navbar"
import HeroSection from "../components/HeroSection"

export default function Home() {
  return (
    <>
      <Navbar />
      <HeroSection>
        {/* <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
          <div className="flex flex-col items-center justify-between">
            <h1 className="text-6xl font-bold">HackTJ 11</h1>
            <h2 className="text-2xl font-bold">April 9-11, 2021</h2>
          </div>
        </div> */}
        <h1 className="text-8xl font-bold mb-6 text-sky-400">Aegis</h1>
        <h3 className="text-2xl max-w-2xl text-center font-semibold text-white">Protect your content from unethical use in AI model training or deepfake generation.</h3>
      </HeroSection>
    </>
  );
}
