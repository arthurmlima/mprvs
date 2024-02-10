----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/29/2019 07:35:14 PM
-- Design Name: 
-- Module Name: wrapper_EB_DEC - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;




entity PM_wrapper_EB_DEC is
generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);
  Port (
clk                        : in std_logic ;
reset                        : in std_logic ;
--------Handshakes da entrada do EB
--P_PM_valid_CF_EB              : in    std_logic; -- indica msg dispon?vel pra enviar ao EB
P_PM_ready_EB_CF              : out   std_logic; -- indica que o EB pode receber a mensagem 
P_PM_stall_in_EB              : in    std_logic ; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
P_PM_stall_out_EB             : inout std_logic;  -- indica que o '''
---------handshakes na sa?da do decoder
P_PM_out_send                 : out  std_logic;
--P_PM_ready_CF_DEC             : in std_logic;
P_PM_stall_out_DEC            : inout std_logic ;
P_PM_stall_in_DEC             : in std_logic ;
-------------------------------------------
--P_PM_in_ack : in std_logic;

P_PM_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0);
P_PM_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0);
P_PM_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0);
P_PM_data_in_step      : in std_logic_vector(n_steps-1 downto 0);
P_PM_data_in_frame     : in std_logic_vector(n_frames-1 downto 0);
P_PM_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0);
P_PM_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0);
P_PM_data_in_fb        : in std_logic;
 
P_PM_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
P_PM_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
P_PM_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
P_PM_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
P_PM_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
P_PM_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
P_PM_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
P_PM_data_out_fb        : out std_logic   ;
--Wrapper_PM_ready_EB_CF  : in std_logic;
P_PM_out_direction      : out std_logic_vector(5 downto 0);
P_PM_ack                : out std_logic;
P_PM_req                : in std_logic
   
  
   );
end PM_wrapper_EB_DEC;

architecture Behavioral of PM_wrapper_EB_DEC is


signal	   w_PM_ready_DEC_EB             :  std_logic; -- sinal indica se o decoder est? pronto para receber a mensagem 

signal     w_PM_x_dest                   :  std_logic_vector(img_width-1 downto 0);
signal     w_PM_y_dest                   :  std_logic_vector(img_height-1 downto 0);
signal     w_PM_x_orig                   :  std_logic_vector(img_width-1 downto 0);
signal     w_PM_y_orig                   :  std_logic_vector(img_height-1 downto 0);
signal     w_PM_fb                       :  std_logic;
signal     w_PM_out_send                 :  std_logic;
signal     w_PM_ready_CF_DEC             :  std_logic;
signal     w_PM_out_direction            :  std_logic_vector(5 downto 0);

SIGnal     w_PM_valid_EB_DEC   :  std_logic;
SIGnal     w_PM_out_new_EB_DEC  :  std_logic;

signal    s_PM_data_pixel     :  std_logic_vector(pix_depth-1 downto 0);  
signal    s_PM_data_x_dest    :  std_logic_vector(img_width-1 downto 0);  
signal    s_PM_data_y_dest    :  std_logic_vector(img_height-1 downto 0); 
signal    s_PM_data_step      :  std_logic_vector(n_steps-1 downto 0);    
signal    s_PM_data_frame     :  std_logic_vector(n_frames-1 downto 0);   
signal    s_PM_data_x_orig    :  std_logic_vector(img_width-1 downto 0);  
signal    s_PM_data_y_orig    :  std_logic_vector(img_height-1 downto 0); 
signal    s_PM_data_fb        :  std_logic   ;                                                                    
signal    s_PM_ack                :  std_logic;                               
signal    s_PM_req                : std_logic;                              





      
begin

Inst_dec : entity work.PM_dec

generic map(
                                    x_init          => x_init       ,
                                    y_init          => y_init       ,
                                    img_width       => img_width    ,
                                    img_height      => img_height   ,
                                    n_frames        => n_frames     ,
                                    n_steps         => n_steps      ,
                                    pix_depth       => pix_depth    ,
                                    subimg_width    => subimg_width ,
                                    subimg_height   => subimg_height,
                                    buffer_length   => buffer_length
                                    )

port map (


clk                   =>  clk ,
rst                   => reset ,



PM_stall_out_DEC      =>   P_PM_stall_out_DEC ,
PM_stall_in_DEC       =>   P_PM_stall_in_DEC  ,

PM_in_x_dest          =>  w_PM_x_dest,
PM_in_y_dest          =>  w_PM_y_dest,
PM_in_x_orig          =>  w_PM_x_orig,
PM_in_y_orig          =>  w_PM_y_orig,
PM_in_fb              =>  w_PM_fb    ,

PM_out_direction      =>  P_PM_out_direction,

PM_out_send           => P_PM_out_send   ,


PM_ready_DEC_EB       => w_PM_ready_DEC_EB,
PM_valid_EB_DEC       => w_PM_valid_EB_DEC,
PM_out_new_EB_DEC     => w_PM_out_new_EB_DEC

);

Inst_elastic_buffer : entity work.PM_elastic_buffer

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    ,
subimg_width    => subimg_width ,
subimg_height   => subimg_height,
buffer_length   => buffer_length
)
port map (


clk                     =>     clk , 
rst                     =>     reset ,       
          
PM_valid_EB_DEC         =>     w_PM_valid_EB_DEC,
PM_ready_DEC_EB         =>     w_PM_ready_DEC_EB,
  
PM_out_new_EB_DEC     => w_PM_out_new_EB_DEC,  

PM_ready_EB_CF          =>  P_PM_ready_EB_CF,
    
PM_stall_in_EB          => P_PM_stall_in_EB,
PM_stall_out_EB         => P_PM_stall_out_EB,

                       
PM_data_in_pixel        =>    s_PM_data_pixel  ,
PM_data_in_x_dest       =>    s_PM_data_x_dest ,
PM_data_in_y_dest       =>    s_PM_data_y_dest ,
PM_data_in_step         =>    s_PM_data_step   ,
PM_data_in_frame        =>    s_PM_data_frame  ,
PM_data_in_x_orig       =>    s_PM_data_x_orig ,
PM_data_in_y_orig       =>    s_PM_data_y_orig ,
PM_data_in_fb           =>    s_PM_data_fb     ,
PM_ack                  =>    s_PM_ack         ,
PM_req                  =>    s_PM_req         ,

PM_data_out_pixel       =>     P_PM_data_out_pixel ,
PM_data_out_x_dest      =>     w_PM_x_dest,
PM_data_out_y_dest      =>     w_PM_y_dest,
PM_data_out_step        =>     P_PM_data_out_step  ,
PM_data_out_frame       =>     P_PM_data_out_frame ,
PM_data_out_x_orig      =>     w_PM_x_orig,
PM_data_out_y_orig      =>     w_PM_y_orig,
PM_data_out_fb          =>     w_PM_fb  
--PM_ready_CF_EB          =>     Wrapper_PM_ready_EB_CF,






   


);


Inst_in_pm_ctrl : entity work.in_pm_controller

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    
)
port map (


clk                     =>     clk , 
reset                     =>     reset ,       
                                 
oc_PM_pixel        =>    P_PM_data_in_pixel  ,
oc_PM_x_dest       =>    P_PM_data_in_x_dest ,
oc_PM_y_dest       =>    P_PM_data_in_y_dest ,
oc_PM_step         =>    P_PM_data_in_step   ,
oc_PM_frame        =>    P_PM_data_in_frame  ,
oc_PM_x_orig       =>    P_PM_data_in_x_orig,
oc_PM_y_orig       =>    P_PM_data_in_y_orig,
oc_PM_fb           =>    P_PM_data_in_fb    ,        
oc_PM_new_msg      =>    P_PM_req,
oc_PM_ack          =>    P_PM_ack, 


    

 i_PM_pixel       =>     s_PM_data_pixel  ,
 i_PM_x_dest      =>     s_PM_data_x_dest ,
 i_PM_y_dest      =>     s_PM_data_y_dest ,
 i_PM_step        =>     s_PM_data_step   ,
 i_PM_frame       =>     s_PM_data_frame  ,
 i_PM_x_orig      =>     s_PM_data_x_orig ,
 i_PM_y_orig      =>     s_PM_data_y_orig ,
 i_PM_fb          =>     s_PM_data_fb     ,
 i_PM_req         =>     s_PM_req         ,
 i_PM_ack         =>     s_PM_ack          




   


);







     P_PM_data_out_x_dest <= w_PM_x_dest;
     P_PM_data_out_y_dest <= w_PM_y_dest;
     
     P_PM_data_out_x_orig <= w_PM_x_orig;
     P_PM_data_out_y_orig <= w_PM_y_orig;
     P_PM_data_out_fb     <= w_PM_fb    ;
     
 

   
end Behavioral;
