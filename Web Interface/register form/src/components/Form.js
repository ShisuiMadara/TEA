import React from 'react'
import bgimg from "../assets/IITJMU.jpg";
import { useForm } from "react-hook-form";

export default function Form(){
    const { register, handleSubmit} = useForm();
    const onSubmit = data => console.log(data);

    return(
        <section>
            <div className='register'>
                <div className='col-1'>
                    <h2>Sign In</h2>
                    <span>One stop solution to Traffic Data Management</span>

                    <form id='form' className='flex flex-col' onSubmit={handleSubmit(onSubmit)}>
                        <input type="text"{...register("name")} placeholder='username'></input>
                        <input type="password" {...register("password")} placeholder='password'></input>
                        <input type="password" {...register("confirm pwd")} placeholder='confirm password'></input>
                        <input type="text" {...register("mail")} placeholder='e-mail address'></input>
                        <input type="number" {...register("mobile")} placeholder='mobile number'></input>
                    
                        <button className='btn'>Sign In</button>
                    </form>
                </div>
                <div className='col-2'>
                    <img src={bgimg} alt = ""/>
                </div>
            </div>
        </section>
    )
}