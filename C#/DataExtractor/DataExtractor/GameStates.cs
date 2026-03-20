using System;

[Serializable]
public class PlayerState
{
    // Position flags
    public float px, py;
    public float pvx, pvy;

    // Resources flags
    public int hp;
    public int maxHp;

    public int soul;
    public int maxSoul;

    //public bool dead;

    // Facing flags
    public bool facingRight;

    // Movement flags
    public bool onGround;
    public bool falling;
    public bool wallJumping;
    public bool jumping;
    public bool doubleJumping;

    public bool dashing;
    public float dashCooldown;

    public bool shadowDashing;
    public float shadowDashCoolDown;

    // Behavior flags
    public bool canFocus;
    public bool focusing;
    public bool invulnerable;

    // Attack flags
    public bool isAttacking;
    public float attackDuration;
    public float attackRecoveryTime;
    public float attackCooldownTime;
    public bool attackForward;
    public bool attackUp;
    public bool attackDown;
    public bool canCast;
    public bool casting;

    // Boss flags:

    // Position flags
    public float bx, by;
    public float bvx, bvy;

    // Boss Resources flags
    public int bossHp;
    public int bossMaxHp;

    // Behavior flags
    //public bool bossIsDead;
    public int bossState;
    public int bossStateAmount;

    // boss scene hash code
    public int bossSceneHash;
}