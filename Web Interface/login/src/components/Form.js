import React from 'react'
import bgimg from "../assets/IITJMU.jpg";


export default function Form(){


    return(
        <section>
            <div className='login'>
                <div className='col-1'>
                    <h2>Login</h2>
                    <div className='username'>
                        <input type="text" placeholder='username'></input>
                    </div>
                    <div className='password'>
                    <input type="password" placeholder='password'></input>
                    </div>
                        
                        
                        <button className='btn'>Log In</button>
                </div>
                <div className='col-2'>
                    <img src={bgimg} alt = ""/>
                </div>
            </div>
        </section>
    )
}